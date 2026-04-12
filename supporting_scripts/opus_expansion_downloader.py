#!/usr/bin/env python3
"""
OPUS Expansion Downloader for ICES cassini/issna iteration 11.

Extension of opus_stratified_downloader.py that:
  1. Loads an exclusion set (existing manifest IDs + test_split + dropped_by_human)
  2. Pre-filters OPUS candidates against the exclusion set BEFORE stratified sampling
  3. Stratifies by body, time-spreads across the mission, preserves >=3-frame bursts
  4. APPENDS new rows to the manifest (existing rows preserved byte-for-byte,
     new rows written after a fresh header-less append)
  5. Refuses to touch any test_split or dropped_by_human ID

Usage:
    python3 opus_expansion_downloader.py \\
        --output-dir /abs/data/cassini/issna \\
        --manifest-path /abs/artifacts/cassini_issna/image_manifest.csv \\
        --test-split-path /abs/artifacts/cassini_issna/test_split.csv \\
        --dropped-path /abs/artifacts/cassini_issna/dropped_by_human.txt \\
        --new-count 750 \\
        --budget-gb 20
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

# Import helpers from the existing script
sys.path.insert(0, str(Path(__file__).parent))
from opus_stratified_downloader import (  # type: ignore
    api_get,
    download_file,
    sha256_file,
    log,
    sanitize_body_name,
    get_candidate_set,
    compute_stratified_allocation,
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


def fetch_candidates_excluding(
    instrument_id: str, target: str, total_available: int,
    exclusion: Set[str], desired_new: int,
) -> List[dict]:
    """
    Paginate through OPUS candidates for a target, DROP any opusid in the
    exclusion set, and collect up to ~3x desired_new candidates (for later
    stratified sub-sampling). Spreads fetches across the full timeline.

    Uses startobs stride sampling to avoid fetching all candidates when the
    target has tens of thousands.
    """
    if desired_new <= 0:
        return []

    # We want to sample the full timeline. Use a stride: target fetching
    # at most ~ max(500, desired_new*4) rows total per target.
    fetch_budget = max(400, desired_new * 4)
    fetch_budget = min(fetch_budget, total_available)

    # stride over the full timeline
    if total_available > fetch_budget:
        stride = total_available / fetch_budget
    else:
        stride = 1.0

    collected: List[dict] = []
    seen: Set[str] = set()
    batch_size = 100
    startobs = 1
    pages_fetched = 0
    max_pages = max(30, fetch_budget // batch_size + 5)

    while len(collected) < fetch_budget and startobs <= total_available and pages_fetched < max_pages:
        limit = min(batch_size, fetch_budget - len(collected))
        try:
            data = api_get("data.json", {
                "instrument": "Cassini ISS",
                "instrumentid": instrument_id,
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
        added_this_page = 0
        for row in page:
            opusid = row[0]
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
            added_this_page += 1

        # Advance by stride
        if stride > 1.0:
            startobs += max(limit, int(limit * stride))
        else:
            startobs += len(page)

        time.sleep(RATE_LIMIT_DELAY)

        if len(page) < limit:
            break

    return collected


def select_time_spread_with_bursts(
    candidates: List[dict], desired: int,
) -> Tuple[List[dict], int]:
    """
    Select `desired` items: 15% reserved for tight bursts, 85% time-spread.
    Returns (selected_list, burst_frames_count).
    """
    if not candidates:
        return [], 0
    if len(candidates) <= desired:
        # Take them all; still try to count bursts inside for logging
        bursts = _find_tight_bursts(candidates, [], len(candidates))
        return list(candidates), len(bursts)

    # Sort by time
    candidates = sorted(candidates, key=lambda x: x.get("time1") or "")

    burst_budget = max(3, int(desired * 0.15))
    spread_count = desired - burst_budget

    # Even time-spread
    if spread_count >= len(candidates):
        spread_sel = list(candidates)
    else:
        indices = sorted(set(int(i * len(candidates) / spread_count) for i in range(spread_count)))
        spread_sel = [candidates[i] for i in indices]

    # Bursts from remaining candidates
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
    """Append a single row (no header rewrite) to the manifest."""
    # File is guaranteed to exist with header in this expansion flow
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", default="cassini")
    parser.add_argument("--instrument", default="issna")
    parser.add_argument("--instrument-id", default="ISSNA")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--test-split-path", required=True)
    parser.add_argument("--dropped-path", required=True)
    parser.add_argument("--new-count", type=int, default=750,
                        help="Allocator cap (planned). May overshoot due to dead rares.")
    parser.add_argument("--max-new", type=int, default=0,
                        help="Hard stop after this many successful new downloads. 0 = unlimited.")
    parser.add_argument("--budget-gb", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)
    test_split_path = Path(args.test_split_path)
    dropped_path = Path(args.dropped_path)

    log("=== OPUS Expansion Downloader (cassini/issna iter 11) ===")
    log(f"Target NEW images: {args.new_count}")
    log(f"Output dir: {output_dir}")
    log(f"Manifest: {manifest_path}")

    # --- Resume cleanup ---
    if output_dir.exists():
        clean_partial_files(output_dir)

    # --- Load exclusion set ---
    manifest_ids = load_id_set_csv(manifest_path, "image_id")
    test_ids = load_id_set_csv(test_split_path, "image_id")
    dropped_ids = load_id_set_txt(dropped_path)

    log(f"Exclusion: manifest={len(manifest_ids)}, test={len(test_ids)}, dropped={len(dropped_ids)}")
    log(f"  test subset of manifest: {test_ids.issubset(manifest_ids)}")
    log(f"  dropped subset of manifest: {dropped_ids.issubset(manifest_ids)}")

    exclusion: Set[str] = manifest_ids | test_ids | dropped_ids
    log(f"Total exclusion set size: {len(exclusion)}")

    # --- Pre-flight ---
    total_candidates, target_counts = get_candidate_set(args.instrument_id)

    per_image_mb = 0.230
    effective_cap = args.max_new if args.max_new else args.new_count
    estimated_gb = (per_image_mb * effective_cap * 1.25) / 1024
    log(f"Estimated download size: {estimated_gb:.3f} GB "
        f"({effective_cap} x {per_image_mb:.3f} MB x 1.25 overhead)")
    log(f"Disk budget: {args.budget_gb} GB")

    stat = os.statvfs(str(output_dir.parent if output_dir.exists() else manifest_path.parent))
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    log(f"Free disk space: {free_gb:.1f} GB")
    log(f"50% of free space: {free_gb * 0.5:.1f} GB")

    if estimated_gb > args.budget_gb:
        log(f"BLOCKED: disk budget. {estimated_gb:.3f} GB > budget {args.budget_gb} GB")
        sys.exit(2)
    if estimated_gb > free_gb * 0.5:
        log(f"BLOCKED: disk budget. {estimated_gb:.3f} GB > 50% free {free_gb:.1f}")
        sys.exit(2)
    log("Pre-flight PASSED.")

    # --- Stratified allocation for NEW images ---
    # Note: target_counts is the OPUS total per body; allocation over them
    # is still valid because we'll pull from untouched candidates.
    allocation = compute_stratified_allocation(target_counts, args.new_count)
    total_allocated = sum(allocation.values())
    log(f"\n=== Stratified Allocation ({total_allocated} planned) ===")
    for target, count in sorted(allocation.items(), key=lambda x: -x[1])[:30]:
        available = target_counts.get(target, 0)
        body_name = sanitize_body_name(target)
        log(f"  {target:30s}: {count:>5d} / {available:>7d}  -> {body_name}/")
    if len(allocation) > 30:
        log(f"  ... ({len(allocation) - 30} more targets)")

    # --- Selection with exclusion ---
    log("\n=== Fetching OPUS candidates with exclusion ===")
    all_selected: List[Tuple[dict, str]] = []
    per_body_stats: Dict[str, dict] = {}
    bursts_preserved_total = 0
    ran_out_bodies: List[str] = []

    for target, desired in sorted(allocation.items(), key=lambda x: -x[1]):
        available = target_counts.get(target, 0)
        log(f"\n-> {target}: want {desired} new (available in OPUS: {available})")

        candidates = fetch_candidates_excluding(
            args.instrument_id, target, available, exclusion, desired,
        )
        log(f"   fetched {len(candidates)} candidates after exclusion filter")

        if len(candidates) < desired:
            log(f"   WARNING: ran out of candidates for {target} "
                f"(wanted {desired}, got {len(candidates)})")
            ran_out_bodies.append(target)

        selected, burst_count = select_time_spread_with_bursts(candidates, desired)
        bursts_preserved_total += burst_count

        per_body_stats[target] = {
            "wanted": desired,
            "available": available,
            "candidates_after_filter": len(candidates),
            "selected": len(selected),
            "bursts_preserved": burst_count,
        }

        for obs in selected:
            all_selected.append((obs, target))

        log(f"   selected {len(selected)} (bursts preserved in selection: {burst_count})")

    # --- Hard disjoint check ---
    selected_ids = set(o["opusid"] for o, _ in all_selected)
    intersect_manifest = selected_ids & manifest_ids
    intersect_test = selected_ids & test_ids
    intersect_dropped = selected_ids & dropped_ids
    log(f"\n=== Disjointness check ===")
    log(f"  selected unique ids: {len(selected_ids)}")
    log(f"  intersect existing manifest: {len(intersect_manifest)}")
    log(f"  intersect test_split:       {len(intersect_test)}")
    log(f"  intersect dropped_by_human: {len(intersect_dropped)}")

    if intersect_manifest or intersect_test or intersect_dropped:
        log("BLOCKED: disjointness violated. Refusing to download.")
        log(f"  sample manifest overlap: {list(intersect_manifest)[:5]}")
        log(f"  sample test overlap:     {list(intersect_test)[:5]}")
        log(f"  sample dropped overlap:  {list(intersect_dropped)[:5]}")
        sys.exit(3)

    log(f"Disjointness OK. Proceeding with {len(all_selected)} downloads.")

    if args.dry_run:
        # Emit a plan JSON to stdout for logging
        plan = {
            "new_count": args.new_count,
            "selected": len(all_selected),
            "bursts_preserved_total": bursts_preserved_total,
            "ran_out_bodies": ran_out_bodies,
            "per_body_stats": per_body_stats,
        }
        log("DRY RUN plan:")
        log(json.dumps(plan, indent=2))
        sys.exit(0)

    # --- Download loop, appending to manifest as we go ---
    downloaded_count = 0
    failed_count = 0
    consecutive_failures = 0
    max_consecutive_failures = 10

    t0 = time.time()
    for obs, target in all_selected:
        if args.max_new and downloaded_count >= args.max_new:
            log(f"Reached --max-new={args.max_new}. Stopping.")
            break
        opusid = obs["opusid"]
        if opusid in manifest_ids:
            # Should never happen, we already filtered
            log(f"  SKIP {opusid}: already in manifest (defensive check)")
            continue

        body_name = sanitize_body_name(target)
        body_dir = output_dir / body_name

        log(f"[{downloaded_count+failed_count+1}/{len(all_selected)}] {opusid} -> {body_name}/")
        image_path, metadata_path = download_observation(opusid, body_dir, args.instrument_id)

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
                "mission": args.mission,
                "instrument": args.instrument,
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
            log(f"  progress: {downloaded_count} ok, {failed_count} fail, "
                f"{dt:.1f}s elapsed")

    # --- Summary ---
    log("\n=== Expansion Download Complete ===")
    log(f"Downloaded: {downloaded_count}")
    log(f"Failed:     {failed_count}")
    log(f"Cumulative manifest size: {len(manifest_ids)}")
    log(f"Bursts preserved in selection plan: {bursts_preserved_total}")
    if ran_out_bodies:
        log(f"Bodies that ran out of candidates: {len(ran_out_bodies)}")
        for b in ran_out_bodies:
            log(f"  - {b}")

    summary = {
        "downloaded": downloaded_count,
        "failed": failed_count,
        "cumulative_manifest": len(manifest_ids),
        "bursts_preserved": bursts_preserved_total,
        "ran_out_bodies": ran_out_bodies,
        "per_body_stats": per_body_stats,
    }
    summary_path = manifest_path.parent / "expansion_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
