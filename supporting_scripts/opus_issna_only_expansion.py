#!/usr/bin/env python3
"""
OPUS ISSNA-only Expansion Downloader - Second expansion for cassini/issna.

Downloads 200 new ISS Narrow Angle Camera images ONLY (co-iss-n* prefix).
Hard-filters out all co-iss-w* (Wide Angle) IDs at every stage.
Stratifies by body for diversity, preserves burst material.

Usage:
    python3 opus_issna_only_expansion.py \
        --output-dir /abs/data/cassini/issna \
        --manifest-path /abs/artifacts/cassini_issna/image_manifest.csv \
        --test-split-path /abs/artifacts/cassini_issna/test_split.csv \
        --dropped-path /abs/artifacts/cassini_issna/dropped_by_human.txt \
        --isswa-excluded-path /abs/artifacts/cassini_issna/isswa_excluded.txt \
        --new-count 200
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

sys.path.insert(0, str(Path(__file__).parent))
from opus_stratified_downloader import (
    api_get,
    download_file,
    sha256_file,
    log,
    sanitize_body_name,
    _find_tight_bursts,
    download_observation,
    clean_partial_files,
    RATE_LIMIT_DELAY,
    NON_BODY_TARGETS,
)

MANIFEST_FIELDS = [
    "image_id", "mission", "instrument", "body", "filter",
    "exposure_s", "timestamp_utc", "target_body_from_metadata",
    "source_url", "sha256", "local_path",
    "instrument_verified", "state_verified",
]

# Priority bodies for diversity - user explicitly asked for spread
PRIORITY_BODIES = [
    "Saturn", "Titan", "Jupiter", "Enceladus", "Iapetus",
    "Rhea", "Dione", "Tethys", "Mimas", "Hyperion",
]


def is_issna(opusid: str) -> bool:
    """True if the OPUS ID belongs to ISS Narrow Angle Camera."""
    return opusid.startswith("co-iss-n")


def is_isswa(opusid: str) -> bool:
    """True if the OPUS ID belongs to ISS Wide Angle Camera."""
    return opusid.startswith("co-iss-w")


def load_id_set_csv(path: Path, col: str) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(col)
            if v:
                out.add(v.strip())
    return out


def load_id_set_txt(path: Path) -> Set[str]:
    out: Set[str] = set()
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.add(line)
    return out


def get_candidate_counts_issna_only() -> Tuple[int, dict]:
    """Get per-target counts for Cassini ISS with instrumentid=ISSNA."""
    log("Querying OPUS for ISSNA candidate set...")
    count_data = api_get("meta/result_count.json", {
        "instrument": "Cassini ISS",
        "instrumentid": "ISSNA",
    })
    total = count_data["data"][0]["result_count"]
    log(f"Total ISSNA candidates in OPUS: {total:,}")

    target_data = api_get("meta/mults/target.json", {
        "instrument": "Cassini ISS",
        "instrumentid": "ISSNA",
    })
    target_counts = target_data.get("mults", {})
    return total, target_counts


def compute_diversity_allocation(
    target_counts: dict, total_cap: int, priority_bodies: List[str],
) -> dict:
    """
    Allocate images across bodies with emphasis on diversity.
    Priority bodies get a minimum guaranteed allocation.
    Non-body targets get a small allocation. Rare bodies get all.
    """
    allocation = {}
    used = 0

    # Phase 1: rare bodies (< 20 candidates) - take all, capped at 5 each
    rare_bodies = {}
    non_body_alloc = {}
    major_bodies = {}

    for target, count in target_counts.items():
        if count == 0:
            continue
        if target.lower() in NON_BODY_TARGETS:
            alloc = min(count, 2)
            non_body_alloc[target] = alloc
        elif count < 20:
            alloc = min(count, 3)  # Small cap for rare bodies in 200-image budget
            rare_bodies[target] = alloc
        else:
            major_bodies[target] = count

    # Phase 2: priority bodies get a minimum floor
    # With 200 images, give each priority body at least 10 images
    priority_floor = 10
    priority_budget = 0
    for body_name in priority_bodies:
        if body_name in major_bodies:
            allocation[body_name] = min(priority_floor, major_bodies[body_name])
            priority_budget += allocation[body_name]

    # Phase 3: non-priority major bodies get a small allocation
    non_priority_major = {t: c for t, c in major_bodies.items()
                          if t not in allocation}

    # Budget remaining after priority floors, rare, and non-body
    rare_total = sum(rare_bodies.values())
    non_body_total = sum(non_body_alloc.values())
    remaining = total_cap - priority_budget - rare_total - non_body_total

    if remaining > 0:
        # Split remaining proportionally among all major bodies (including priority)
        all_major = {}
        for t, c in major_bodies.items():
            if t in allocation:
                # Priority body: can get more above the floor
                all_major[t] = c
            else:
                all_major[t] = c

        total_major_pool = sum(all_major.values())
        if total_major_pool > 0:
            for t, c in all_major.items():
                extra = max(1, int(remaining * c / total_major_pool))
                extra = min(extra, c)
                if t in allocation:
                    allocation[t] = min(allocation[t] + extra, c)
                else:
                    allocation[t] = min(extra, c)

    # Add rare and non-body
    for t, a in rare_bodies.items():
        allocation[t] = a
    for t, a in non_body_alloc.items():
        allocation[t] = a

    # Trim to cap
    current = sum(allocation.values())
    if current > total_cap:
        # Scale down non-priority major bodies first
        excess = current - total_cap
        scalable = {t: a for t, a in allocation.items()
                    if t not in PRIORITY_BODIES
                    and t.lower() not in NON_BODY_TARGETS
                    and target_counts.get(t, 0) >= 20}
        scalable_total = sum(scalable.values())
        if scalable_total > 0 and excess > 0:
            for t in scalable:
                reduction = max(0, int(excess * scalable[t] / scalable_total))
                allocation[t] = max(1, allocation[t] - reduction)

    # Final trim if still over
    current = sum(allocation.values())
    if current > total_cap:
        # Remove smallest allocations first
        sorted_allocs = sorted(allocation.items(), key=lambda x: x[1])
        for t, a in sorted_allocs:
            if current <= total_cap:
                break
            if t not in [p for p in PRIORITY_BODIES] and a > 0:
                remove = min(a - 1, current - total_cap) if a > 1 else 0
                if remove > 0:
                    allocation[t] -= remove
                    current -= remove

    return {t: a for t, a in allocation.items() if a > 0}


def fetch_candidates_issna_only(
    target: str, total_available: int,
    exclusion: Set[str], desired_new: int,
) -> List[dict]:
    """
    Fetch OPUS candidates for a target, hard-filtering:
    1. Only co-iss-n* IDs (ISSNA)
    2. Not in exclusion set
    """
    if desired_new <= 0:
        return []

    fetch_budget = max(400, desired_new * 5)
    fetch_budget = min(fetch_budget, total_available)

    collected: List[dict] = []
    seen: Set[str] = set()
    batch_size = 100
    startobs = 1
    pages_fetched = 0
    max_pages = max(40, fetch_budget // batch_size + 5)
    isswa_filtered = 0

    while len(collected) < fetch_budget and startobs <= total_available and pages_fetched < max_pages:
        limit = min(batch_size, fetch_budget - len(collected) + 50)  # overfetch slightly
        try:
            data = api_get("data.json", {
                "instrument": "Cassini ISS",
                "instrumentid": "ISSNA",
                "target": target,
                "limit": limit,
                "startobs": startobs,
                "cols": "opusid,target,time1,COISSfilter,observationduration",
            })
        except Exception as e:
            log(f"  OPUS fetch failed for {target} at startobs={startobs}: {e}")
            break

        pages_fetched += 1
        if not data or not data.get("page"):
            break

        page = data["page"]
        for row in page:
            opusid = row[0]
            # CRITICAL: hard filter for ISSNA only
            if is_isswa(opusid):
                isswa_filtered += 1
                continue
            if not is_issna(opusid):
                # Neither ISSNA nor ISSWA -- skip unknown prefix
                continue
            if opusid in exclusion or opusid in seen:
                continue
            seen.add(opusid)
            collected.append({
                "opusid": opusid,
                "target": row[1],
                "time1": row[2],
                "filter": row[3],
                "duration": row[4],
            })

        # Stride through timeline
        if total_available > fetch_budget:
            stride = total_available / fetch_budget
            startobs += max(len(page), int(len(page) * stride))
        else:
            startobs += len(page)

        time.sleep(RATE_LIMIT_DELAY)

        if len(page) < limit:
            break

    if isswa_filtered > 0:
        log(f"  Filtered out {isswa_filtered} ISSWA IDs from OPUS results")

    return collected


def select_time_spread_with_bursts(
    candidates: List[dict], desired: int,
) -> Tuple[List[dict], int]:
    """Select desired items: 15% bursts, 85% time-spread."""
    if not candidates:
        return [], 0
    if len(candidates) <= desired:
        return list(candidates), 0

    candidates = sorted(candidates, key=lambda x: x.get("time1") or "")

    burst_budget = max(3, int(desired * 0.15))
    spread_count = desired - burst_budget

    if spread_count >= len(candidates):
        spread_sel = list(candidates)
    else:
        indices = sorted(set(
            int(i * len(candidates) / spread_count) for i in range(spread_count)
        ))
        spread_sel = [candidates[i] for i in indices]

    burst_sel = _find_tight_bursts(candidates, spread_sel, burst_budget)

    sel_ids = set(o["opusid"] for o in spread_sel)
    out = list(spread_sel)
    burst_added = 0
    for o in burst_sel:
        if o["opusid"] not in sel_ids:
            out.append(o)
            sel_ids.add(o["opusid"])
            burst_added += 1
        if len(out) >= desired:
            break

    return out[:desired], burst_added


def append_manifest_row(manifest_path: Path, row: dict):
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--test-split-path", required=True)
    parser.add_argument("--dropped-path", required=True)
    parser.add_argument("--isswa-excluded-path", required=True)
    parser.add_argument("--new-count", type=int, default=200)
    parser.add_argument("--budget-gb", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)

    log("=== OPUS ISSNA-Only Expansion Downloader (200 new images) ===")
    log(f"Target NEW images: {args.new_count}")
    log(f"CRITICAL: Only co-iss-n* IDs will be downloaded. Zero co-iss-w* tolerance.")

    # --- Resume cleanup ---
    if output_dir.exists():
        clean_partial_files(output_dir)

    # --- Load all exclusion sets ---
    manifest_ids = load_id_set_csv(Path(args.manifest_path), "image_id")
    test_ids = load_id_set_csv(Path(args.test_split_path), "image_id")
    dropped_ids = load_id_set_txt(Path(args.dropped_path))
    isswa_ids = load_id_set_txt(Path(args.isswa_excluded_path))

    log(f"Exclusion sets: manifest={len(manifest_ids)}, test={len(test_ids)}, "
        f"dropped={len(dropped_ids)}, isswa_excluded={len(isswa_ids)}")

    exclusion = manifest_ids | test_ids | dropped_ids | isswa_ids
    log(f"Total exclusion set: {len(exclusion)} IDs")

    # --- Pre-flight ---
    total_candidates, target_counts = get_candidate_counts_issna_only()

    per_image_mb = 0.230
    estimated_gb = (per_image_mb * args.new_count * 1.25) / 1024
    log(f"Estimated download size: {estimated_gb:.4f} GB "
        f"({args.new_count} x {per_image_mb} MB x 1.25)")
    log(f"Disk budget: {args.budget_gb} GB")

    stat = os.statvfs(str(output_dir if output_dir.exists() else Path(args.manifest_path).parent))
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    log(f"Free disk space: {free_gb:.1f} GB")
    log(f"50% of free: {free_gb * 0.5:.1f} GB")

    if estimated_gb > args.budget_gb:
        log(f"BLOCKED: disk budget exceeded")
        sys.exit(2)
    if estimated_gb > free_gb * 0.5:
        log(f"BLOCKED: exceeds 50% free space")
        sys.exit(2)
    log("Pre-flight PASSED.")

    # --- Diversity-focused allocation ---
    allocation = compute_diversity_allocation(target_counts, args.new_count, PRIORITY_BODIES)
    total_allocated = sum(allocation.values())
    log(f"\n=== Diversity Allocation ({total_allocated} planned across {len(allocation)} bodies) ===")
    for target, count in sorted(allocation.items(), key=lambda x: -x[1])[:40]:
        available = target_counts.get(target, 0)
        body_name = sanitize_body_name(target)
        is_prio = " [PRIORITY]" if target in PRIORITY_BODIES else ""
        log(f"  {target:25s}: {count:>4d} / {available:>7d}  -> {body_name}/{is_prio}")
    if len(allocation) > 40:
        log(f"  ... ({len(allocation) - 40} more)")

    # --- Fetch candidates and select ---
    log("\n=== Fetching OPUS candidates (ISSNA-only filter active) ===")
    all_selected: List[Tuple[dict, str]] = []
    per_body_stats: Dict[str, dict] = {}
    isswa_total_filtered = 0

    for target, desired in sorted(allocation.items(), key=lambda x: -x[1]):
        available = target_counts.get(target, 0)
        log(f"\n-> {target}: want {desired} new (OPUS total: {available})")

        candidates = fetch_candidates_issna_only(target, available, exclusion, desired)

        # Double-check: assert all candidates are co-iss-n*
        isswa_in_candidates = [c for c in candidates if not is_issna(c["opusid"])]
        if isswa_in_candidates:
            log(f"  CRITICAL: {len(isswa_in_candidates)} non-ISSNA IDs in candidates! Removing.")
            candidates = [c for c in candidates if is_issna(c["opusid"])]
            isswa_total_filtered += len(isswa_in_candidates)

        log(f"   {len(candidates)} ISSNA candidates after exclusion")

        selected, burst_count = select_time_spread_with_bursts(candidates, desired)

        per_body_stats[target] = {
            "wanted": desired,
            "available": available,
            "candidates_after_filter": len(candidates),
            "selected": len(selected),
            "bursts_preserved": burst_count,
        }

        for obs in selected:
            all_selected.append((obs, target))

        log(f"   selected {len(selected)} (bursts: {burst_count})")

    # --- Triple-check disjointness and ISSNA-only ---
    selected_ids = set(o["opusid"] for o, _ in all_selected)

    non_issna = [sid for sid in selected_ids if not is_issna(sid)]
    if non_issna:
        log(f"BLOCKED: {len(non_issna)} non-ISSNA IDs in final selection: {non_issna[:5]}")
        sys.exit(3)

    overlap_manifest = selected_ids & manifest_ids
    overlap_test = selected_ids & test_ids
    overlap_dropped = selected_ids & dropped_ids
    overlap_isswa = selected_ids & isswa_ids

    log(f"\n=== Final Checks ===")
    log(f"  Selected unique IDs: {len(selected_ids)}")
    log(f"  All co-iss-n* prefix: {all(is_issna(s) for s in selected_ids)}")
    log(f"  Overlap with manifest: {len(overlap_manifest)}")
    log(f"  Overlap with test_split: {len(overlap_test)}")
    log(f"  Overlap with dropped: {len(overlap_dropped)}")
    log(f"  Overlap with isswa_excluded: {len(overlap_isswa)}")

    if overlap_manifest or overlap_test or overlap_dropped or overlap_isswa:
        log("BLOCKED: disjointness violated!")
        sys.exit(3)

    log(f"All checks PASSED. Proceeding with {len(all_selected)} downloads.")

    if args.dry_run:
        plan = {
            "new_count": args.new_count,
            "selected": len(all_selected),
            "per_body_stats": per_body_stats,
            "all_issna": all(is_issna(s) for s in selected_ids),
        }
        log("DRY RUN:")
        log(json.dumps(plan, indent=2))
        sys.exit(0)

    # --- Download ---
    downloaded_count = 0
    failed_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 10

    t0 = time.time()
    for obs, target in all_selected:
        opusid = obs["opusid"]

        # Final safety: refuse ISSWA
        if not is_issna(opusid):
            log(f"  REFUSE {opusid}: not ISSNA")
            continue

        if opusid in manifest_ids:
            log(f"  SKIP {opusid}: already in manifest")
            continue

        body_name = sanitize_body_name(target)
        body_dir = output_dir / body_name

        log(f"[{downloaded_count+failed_count+1}/{len(all_selected)}] {opusid} -> {body_name}/")
        image_path, metadata_path = download_observation(opusid, body_dir, "ISSNA")

        if image_path and image_path.exists():
            img_hash = sha256_file(image_path)
            filter_name = obs.get("filter", "") or ""
            exposure = obs.get("duration", "") or ""
            timestamp = obs.get("time1", "") or ""

            try:
                img_data = api_get(f"image/full/{opusid}.json")
                source_url = img_data["data"][0]["url"] if img_data.get("data") else ""
            except Exception:
                source_url = ""

            row = {
                "image_id": opusid,
                "mission": "cassini",
                "instrument": "issna",
                "body": body_name,
                "filter": filter_name,
                "exposure_s": exposure,
                "timestamp_utc": timestamp,
                "target_body_from_metadata": target,
                "source_url": source_url,
                "sha256": img_hash,
                "local_path": str(image_path),
                "instrument_verified": "true",
                "state_verified": "false",
            }
            append_manifest_row(manifest_path, row)
            manifest_ids.add(opusid)
            downloaded_count += 1
            consecutive_failures = 0
        else:
            failed_count += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                log(f"ERROR: {max_consecutive_failures} consecutive failures. Stopping.")
                break

        if (downloaded_count + failed_count) % 25 == 0:
            dt = time.time() - t0
            log(f"  progress: {downloaded_count} ok, {failed_count} fail, {dt:.1f}s elapsed")

    # --- Summary ---
    log(f"\n=== ISSNA-Only Expansion Complete ===")
    log(f"Downloaded: {downloaded_count}")
    log(f"Failed: {failed_count}")
    log(f"Cumulative manifest: {len(manifest_ids)}")
    log(f"Zero ISSWA contamination: {downloaded_count > 0 and all(is_issna(o['opusid']) for o, _ in all_selected)}")

    # Per-body breakdown of what was downloaded
    body_downloaded = defaultdict(int)
    # Re-read last N rows of manifest to get the new ones
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)
    new_rows = all_rows[-(downloaded_count):]  if downloaded_count > 0 else []
    for row in new_rows:
        body_downloaded[row["body"]] += 1

    log(f"\nPer-body breakdown (new downloads):")
    for body, count in sorted(body_downloaded.items(), key=lambda x: -x[1]):
        log(f"  {body:25s}: {count}")

    summary = {
        "downloaded": downloaded_count,
        "failed": failed_count,
        "cumulative_manifest": len(manifest_ids),
        "isswa_contamination": 0,
        "per_body_downloaded": dict(body_downloaded),
        "per_body_stats": per_body_stats,
    }
    summary_path = Path(args.manifest_path).parent / "issna_expansion2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
