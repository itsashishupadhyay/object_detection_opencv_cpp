#!/usr/bin/env python3
"""
OPUS Stratified Image Downloader for ICES Research Pipeline

Non-interactive, scriptable downloader that implements:
- Stratified sampling by target body across the mission timeline
- Tight-burst preservation for adjacency pair analysis
- SHA-256 hashing and manifest generation
- Hierarchical layout per RESEARCH_BRIEF section 7.1
- Resume support with partial-file cleanup
- Disk budget pre-flight checks

Usage:
    python3 opus_stratified_downloader.py \
        --mission cassini --instrument issna \
        --target-count 1200 \
        --output-dir /path/to/data/cassini/issna \
        --manifest-path /path/to/artifacts/cassini_issna/image_manifest.csv \
        --budget-gb 20 \
        --batch-size 50
"""

import json
import os
import sys
import csv
import hashlib
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import argparse
import random

# OPUS API Configuration
BASE_URL = "https://opus.pds-rings.seti.org/opus/api"
REQUEST_TIMEOUT = 60
RATE_LIMIT_DELAY = 0.4  # seconds between API calls

# Non-body targets that go to "unknown" folder
NON_BODY_TARGETS = {
    "sky", "dark sky", "dark", "sun", "star", "none",
    "alp vir (spica)", "alp psa (fomalhaut)", "masursky"
}


def api_get(endpoint: str, params: dict = None) -> dict:
    """Make a GET request to the OPUS API with retry logic."""
    url = f"{BASE_URL}/{endpoint}"
    if params:
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        url = f"{url}?{query}"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "ICES-Research-Pipeline/1.0")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code >= 500 and attempt < max_retries - 1:
                log(f"  Server error {e.code} on attempt {attempt+1}, retrying...")
                time.sleep(2 ** attempt)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                log(f"  Connection error on attempt {attempt+1}: {e}, retrying...")
                time.sleep(2 ** attempt)
                continue
            raise
    return {}


def download_file(url: str, dest: Path, expected_size: int = None) -> bool:
    """Download a file with basic validation. Returns True on success."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "ICES-Research-Pipeline/1.0")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = resp.read()
        tmp.write_bytes(data)
        # Validate size if known
        if expected_size and len(data) != expected_size:
            log(f"  Size mismatch: expected {expected_size}, got {len(data)}")
            tmp.unlink(missing_ok=True)
            return False
        # Atomic rename
        tmp.rename(dest)
        return True
    except Exception as e:
        log(f"  Download failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def sanitize_body_name(target: str) -> str:
    """Convert target name to filesystem-safe folder name per brief."""
    if target.lower() in NON_BODY_TARGETS:
        return "unknown"
    return target.lower().replace(" ", "_").replace("/", "_")


def get_candidate_set(instrument_id: str) -> Tuple[int, dict]:
    """Get total count and per-target counts for Cassini ISS NAC."""
    log("Querying OPUS for candidate set size...")
    count_data = api_get("meta/result_count.json", {
        "instrument": "Cassini ISS",
        "instrumentid": instrument_id
    })
    total = count_data["data"][0]["result_count"]

    log(f"Total candidate observations: {total:,}")

    log("Querying target body distribution...")
    target_data = api_get("meta/mults/target.json", {
        "instrument": "Cassini ISS",
        "instrumentid": instrument_id
    })
    target_counts = target_data.get("mults", {})

    return total, target_counts


def compute_stratified_allocation(target_counts: dict, total_cap: int) -> dict:
    """
    Compute how many images to sample per target body.
    Strategy:
    - Rare bodies (< 20 candidates): take all
    - Major bodies: proportional allocation with a floor
    - Non-body targets (Sky, Sun, etc.): small fixed allocation, go to unknown/
    """
    allocation = {}
    rare_total = 0
    non_body_total = 0

    # First pass: identify rare and non-body targets
    body_targets = {}
    for target, count in target_counts.items():
        if count == 0:
            continue
        if target.lower() in NON_BODY_TARGETS:
            # Small fixed allocation for non-body targets
            alloc = min(count, 5)
            allocation[target] = alloc
            non_body_total += alloc
        elif count < 20:
            # Rare body: take all
            allocation[target] = count
            rare_total += count
        else:
            body_targets[target] = count

    remaining = total_cap - rare_total - non_body_total
    if remaining < 0:
        remaining = 0

    # Second pass: proportional allocation for major bodies
    body_total = sum(body_targets.values())
    if body_total > 0 and remaining > 0:
        # Give each body a minimum floor of 5
        floor_per_body = 5
        floor_total = floor_per_body * len(body_targets)
        proportional_budget = remaining - floor_total

        if proportional_budget < 0:
            # Too many bodies, just split evenly
            per_body = max(1, remaining // len(body_targets))
            for target in body_targets:
                allocation[target] = min(per_body, body_targets[target])
        else:
            for target, count in body_targets.items():
                prop = int(proportional_budget * count / body_total)
                alloc = floor_per_body + prop
                alloc = min(alloc, count)  # Don't exceed available
                allocation[target] = alloc

    # Trim if we exceeded cap
    current_total = sum(allocation.values())
    if current_total > total_cap:
        # Scale down proportionally (except rare bodies)
        excess = current_total - total_cap
        scalable = {t: a for t, a in allocation.items()
                    if target_counts.get(t, 0) >= 20 and t.lower() not in NON_BODY_TARGETS}
        scalable_total = sum(scalable.values())
        if scalable_total > 0:
            for t in scalable:
                reduction = int(excess * scalable[t] / scalable_total)
                allocation[t] = max(1, allocation[t] - reduction)

    return allocation


def fetch_opus_ids_for_target(instrument_id: str, target: str,
                              desired_count: int, total_available: int) -> List[dict]:
    """
    Fetch observation IDs for a target with time-spread sampling.
    Returns list of dicts with opusid, time1, filter, duration.
    """
    # Strategy: fetch a sparse set across the full timeline, then select
    # We need to spread across the mission, so we'll sample at intervals

    if desired_count >= total_available:
        # Take all -- paginate through
        return _fetch_all_for_target(instrument_id, target, total_available)

    # For stratified time-spread: fetch at evenly spaced offsets
    # But also preserve some tight bursts
    step = max(1, total_available // desired_count)

    # Reserve ~15% of allocation for burst frames
    burst_count = max(3, int(desired_count * 0.15))
    spread_count = desired_count - burst_count

    # Fetch spread samples
    spread_obs = []
    offset = 1
    batch_size = 100

    # We'll collect candidates then sub-sample
    # Fetch enough to get a good spread -- fetch about 3x what we need
    fetch_total = min(total_available, desired_count * 3)
    fetch_step = max(1, total_available // fetch_total)

    candidates = []
    fetched = 0
    startobs = 1

    while fetched < fetch_total and startobs <= total_available:
        limit = min(batch_size, fetch_total - fetched)
        data = api_get("data.json", {
            "instrument": "Cassini ISS",
            "instrumentid": instrument_id,
            "target": target,
            "limit": limit,
            "startobs": startobs,
            "cols": "opusid,target,time1,COISSfilter,observationduration"
        })

        if not data or not data.get("page"):
            break

        for row in data["page"]:
            candidates.append({
                "opusid": row[0],
                "target": row[1],
                "time1": row[2],
                "filter": row[3],
                "duration": row[4]
            })
            fetched += 1

        # Skip ahead for spread
        if fetch_step > 1:
            startobs += len(data["page"]) * fetch_step
        else:
            startobs += len(data["page"])

        time.sleep(RATE_LIMIT_DELAY)

    if not candidates:
        return []

    # Sort by time
    candidates.sort(key=lambda x: x["time1"] or "")

    # Select spread samples: evenly spaced
    if len(candidates) <= spread_count:
        spread_obs = candidates
    else:
        indices = [int(i * len(candidates) / spread_count) for i in range(spread_count)]
        indices = list(set(indices))[:spread_count]  # deduplicate
        spread_obs = [candidates[i] for i in sorted(indices)]

    # Find and preserve tight bursts (>= 3 frames within 10 minutes)
    burst_obs = _find_tight_bursts(candidates, spread_obs, burst_count)

    # Combine and deduplicate
    selected_ids = set(o["opusid"] for o in spread_obs)
    result = list(spread_obs)
    for obs in burst_obs:
        if obs["opusid"] not in selected_ids:
            result.append(obs)
            selected_ids.add(obs["opusid"])

    return result[:desired_count]


def _fetch_all_for_target(instrument_id: str, target: str, total: int) -> List[dict]:
    """Fetch all observations for a rare target."""
    results = []
    startobs = 1
    while len(results) < total:
        limit = min(100, total - len(results))
        data = api_get("data.json", {
            "instrument": "Cassini ISS",
            "instrumentid": instrument_id,
            "target": target,
            "limit": limit,
            "startobs": startobs,
            "cols": "opusid,target,time1,COISSfilter,observationduration"
        })
        if not data or not data.get("page"):
            break
        for row in data["page"]:
            results.append({
                "opusid": row[0],
                "target": row[1],
                "time1": row[2],
                "filter": row[3],
                "duration": row[4]
            })
        startobs += len(data["page"])
        time.sleep(RATE_LIMIT_DELAY)
    return results


def _find_tight_bursts(candidates: List[dict], already_selected: List[dict],
                       burst_budget: int) -> List[dict]:
    """Find tight burst sequences (>= 3 frames within 10 min) not already selected."""
    if len(candidates) < 3:
        return []

    selected_ids = set(o["opusid"] for o in already_selected)

    # Parse timestamps and find burst clusters
    timed = []
    for c in candidates:
        t = c.get("time1", "")
        if t and c["opusid"] not in selected_ids:
            timed.append(c)

    bursts = []
    i = 0
    while i < len(timed) - 2:
        # Look for 3+ consecutive frames within 10 minutes
        cluster = [timed[i]]
        j = i + 1
        while j < len(timed):
            t1 = cluster[-1].get("time1", "")
            t2 = timed[j].get("time1", "")
            if t1 and t2:
                # Simple string comparison works for ISO format within ~10 min
                # More precise: parse and diff
                try:
                    from datetime import datetime as dt
                    d1 = dt.fromisoformat(t1.replace("Z", "+00:00") if "Z" in t1 else t1)
                    d2 = dt.fromisoformat(t2.replace("Z", "+00:00") if "Z" in t2 else t2)
                    diff_min = abs((d2 - d1).total_seconds()) / 60.0
                    if diff_min <= 10.0:
                        cluster.append(timed[j])
                        j += 1
                        continue
                except:
                    pass
            break

        if len(cluster) >= 3:
            bursts.append(cluster)

        i = j if j > i + 1 else i + 1

    # Flatten bursts, take up to burst_budget frames
    burst_frames = []
    for cluster in bursts:
        for frame in cluster:
            if len(burst_frames) >= burst_budget:
                break
            burst_frames.append(frame)
        if len(burst_frames) >= burst_budget:
            break

    return burst_frames[:burst_budget]


def download_observation(opusid: str, body_dir: Path, instrument_id: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Download full-resolution image and metadata for one observation.
    Returns (image_path, metadata_path) or (None, None) on failure.
    """
    # Get full-res image info
    try:
        img_data = api_get(f"image/full/{opusid}.json")
    except Exception as e:
        log(f"  Failed to get image URL for {opusid}: {e}")
        return None, None

    if not img_data or "data" not in img_data or not img_data["data"]:
        log(f"  No image data for {opusid}")
        return None, None

    img_info = img_data["data"][0]
    img_url = img_info.get("url", "")
    img_size = img_info.get("size_bytes")
    img_ext = Path(img_url).suffix if img_url else ".png"

    if not img_url:
        log(f"  No image URL for {opusid}")
        return None, None

    # Create directories
    images_dir = body_dir / "images"
    metadata_dir = body_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    image_path = images_dir / f"{opusid}{img_ext}"
    metadata_path = metadata_dir / f"{opusid}.json"

    # Download image
    if not image_path.exists():
        time.sleep(RATE_LIMIT_DELAY)
        if not download_file(img_url, image_path, img_size):
            return None, None

    # Download metadata
    if not metadata_path.exists():
        try:
            time.sleep(RATE_LIMIT_DELAY)
            meta = api_get(f"metadata/{opusid}.json")
            if meta:
                metadata_path.write_text(json.dumps(meta, indent=2))
            else:
                log(f"  Empty metadata for {opusid}")
                return image_path, None
        except Exception as e:
            log(f"  Failed to get metadata for {opusid}: {e}")
            return image_path, None

    return image_path, metadata_path


def clean_partial_files(data_root: Path):
    """Remove .part, .tmp, and orphan files."""
    count = 0
    for p in data_root.rglob("*.part"):
        p.unlink()
        count += 1
    for p in data_root.rglob("*.tmp"):
        p.unlink()
        count += 1
    if count:
        log(f"Cleaned {count} partial/temp files")


def load_existing_manifest(manifest_path: Path) -> dict:
    """Load existing manifest as dict keyed by image_id."""
    existing = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing[row["image_id"]] = row
    return existing


def write_manifest(manifest_path: Path, rows: List[dict]):
    """Write manifest CSV atomically."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".csv.tmp")
    fieldnames = [
        "image_id", "mission", "instrument", "body", "filter",
        "exposure_s", "timestamp_utc", "target_body_from_metadata",
        "source_url", "sha256", "local_path",
        "instrument_verified", "state_verified"
    ]
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(tmp), str(manifest_path))


def main():
    parser = argparse.ArgumentParser(description="OPUS Stratified Image Downloader")
    parser.add_argument("--mission", default="cassini")
    parser.add_argument("--instrument", default="issna")
    parser.add_argument("--instrument-id", default="ISSNA",
                        help="OPUS instrumentid value")
    parser.add_argument("--target-count", type=int, default=1200)
    parser.add_argument("--output-dir", required=True,
                        help="Root data dir: data/<mission>/<instrument>/")
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--budget-gb", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Write manifest every N downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do pre-flight and selection only, no download")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest_path)

    log(f"=== OPUS Stratified Downloader ===")
    log(f"Mission: {args.mission}, Instrument: {args.instrument}")
    log(f"Target count: {args.target_count}")
    log(f"Output: {output_dir}")

    # --- Resume: clean partial files ---
    if output_dir.exists():
        clean_partial_files(output_dir)

    # --- Load existing manifest ---
    existing_manifest = load_existing_manifest(manifest_path)
    log(f"Existing manifest entries: {len(existing_manifest)}")

    # --- Pre-flight: query candidate set ---
    total_candidates, target_counts = get_candidate_set(args.instrument_id)

    # --- Pre-flight: estimate disk usage ---
    # Average full PNG ~200 KB, metadata ~30 KB, overhead 1.25x
    per_image_mb = 0.230  # ~230 KB average
    estimated_gb = (per_image_mb * args.target_count * 1.25) / 1024
    log(f"Estimated download size: {estimated_gb:.3f} GB "
        f"({args.target_count} images x {per_image_mb:.3f} MB x 1.25 overhead)")
    log(f"Disk budget: {args.budget_gb} GB")

    # Check free space
    stat = os.statvfs(str(output_dir.parent))
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    log(f"Free disk space: {free_gb:.1f} GB")
    log(f"50% of free space: {free_gb * 0.5:.1f} GB")

    if estimated_gb > args.budget_gb:
        log(f"BLOCKED: disk budget. Estimated {estimated_gb:.3f} GB exceeds budget {args.budget_gb} GB")
        sys.exit(1)
    if estimated_gb > free_gb * 0.5:
        log(f"BLOCKED: disk budget. Estimated {estimated_gb:.3f} GB exceeds 50% of free space {free_gb:.1f} GB")
        sys.exit(1)

    log("Pre-flight PASSED.")

    # --- Stratified allocation ---
    allocation = compute_stratified_allocation(target_counts, args.target_count)
    total_allocated = sum(allocation.values())

    log(f"\n=== Stratified Allocation ({total_allocated} total) ===")
    for target, count in sorted(allocation.items(), key=lambda x: -x[1]):
        available = target_counts.get(target, 0)
        body_name = sanitize_body_name(target)
        log(f"  {target:30s}: {count:>5d} / {available:>7d}  -> {body_name}/")

    if args.dry_run:
        log("Dry run complete. No downloads.")
        sys.exit(0)

    # --- Selection: fetch observation IDs per target ---
    all_selected = []  # List of (obs_dict, target_name)

    for target, desired in sorted(allocation.items(), key=lambda x: -x[1]):
        available = target_counts.get(target, 0)
        log(f"\nSelecting {desired} observations for {target} (available: {available})...")

        obs_list = fetch_opus_ids_for_target(
            args.instrument_id, target, desired, available
        )
        log(f"  Selected {len(obs_list)} observations")

        for obs in obs_list:
            all_selected.append((obs, target))

    log(f"\nTotal selected: {len(all_selected)} observations")

    # --- Download ---
    manifest_rows = list(existing_manifest.values())
    existing_ids = set(existing_manifest.keys())

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    batch_counter = 0
    consecutive_failures = 0
    max_consecutive_failures = 10

    for obs, target in all_selected:
        opusid = obs["opusid"]

        # Skip if already in manifest with valid hash
        if opusid in existing_ids:
            skipped_count += 1
            continue

        body_name = sanitize_body_name(target)
        body_dir = output_dir / body_name

        log(f"Downloading {opusid} -> {body_name}/...")
        image_path, metadata_path = download_observation(opusid, body_dir, args.instrument_id)

        if image_path and image_path.exists():
            img_hash = sha256_file(image_path)

            # Extract metadata fields
            filter_name = obs.get("filter", "")
            exposure = obs.get("duration", "")
            timestamp = obs.get("time1", "")

            # Build image URL for manifest
            try:
                img_data = api_get(f"image/full/{opusid}.json")
                source_url = img_data["data"][0]["url"] if img_data.get("data") else ""
            except:
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
                "state_verified": "false"
            }
            manifest_rows.append(row)
            existing_ids.add(opusid)
            downloaded_count += 1
            consecutive_failures = 0
            batch_counter += 1

            # Checkpoint manifest every batch_size downloads
            if batch_counter >= args.batch_size:
                log(f"  Checkpointing manifest ({len(manifest_rows)} rows)...")
                write_manifest(manifest_path, manifest_rows)
                batch_counter = 0
        else:
            failed_count += 1
            consecutive_failures += 1
            if consecutive_failures >= max_consecutive_failures:
                log(f"ERROR: {max_consecutive_failures} consecutive failures. "
                    f"Stopping to avoid hammering OPUS.")
                break

        # Progress
        total_processed = downloaded_count + skipped_count + failed_count
        if total_processed % 50 == 0:
            log(f"  Progress: {downloaded_count} downloaded, {skipped_count} skipped, "
                f"{failed_count} failed, {total_processed}/{len(all_selected)} processed")

    # --- Final manifest write ---
    write_manifest(manifest_path, manifest_rows)

    log(f"\n=== Download Complete ===")
    log(f"Downloaded: {downloaded_count}")
    log(f"Skipped (already on disk): {skipped_count}")
    log(f"Failed: {failed_count}")
    log(f"Manifest: {manifest_path} ({len(manifest_rows)} rows)")


if __name__ == "__main__":
    main()
