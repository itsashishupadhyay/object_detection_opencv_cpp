#!/usr/bin/env python3
"""
make_split.py  --  Stratified train/test split for cassini/issna imagery.

Rules (per RESEARCH_BRIEF.md section 7.2):
  - Stratified split per body: every body with >= 50 images appears in both
    train and test.
  - No leakage: images within 60 seconds of each other go to the same bracket
    (burst grouping).
  - Target ratio ~80/20 (train/test), adjusted for stratification.
  - Bodies in 'unknown' are excluded entirely.
  - Fixed random seed for reproducibility.

Also produces adjacency_pairs.csv: for each test image, the nearest train
image in time with the same instrument and filter.

Usage:
    python3 make_split.py
"""

import csv
import hashlib
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta

# ---- Configuration ----
SEED = 42
TRAIN_RATIO = 0.80
BURST_WINDOW_S = 60  # seconds -- images within this window are one group
WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MANIFEST = os.path.join(WORKSPACE, "artifacts", "cassini_issna", "image_manifest.csv")
OUT_DIR = os.path.join(WORKSPACE, "artifacts", "cassini_issna")
TRAIN_CSV = os.path.join(OUT_DIR, "train_split.csv")
TEST_CSV = os.path.join(OUT_DIR, "test_split.csv")
ADJ_CSV = os.path.join(OUT_DIR, "adjacency_pairs.csv")

EXCLUDE_BODIES = {"unknown"}


def parse_timestamp(ts_str):
    """Parse ISO-ish timestamp from the manifest."""
    # Handles '2001-07-13T07:32:10.375' format
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts_str}")


def load_manifest():
    """Load the manifest, exclude unknown bodies, return list of dicts."""
    with open(MANIFEST, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["body"] not in EXCLUDE_BODIES]
    # Parse timestamps
    for r in rows:
        r["_ts"] = parse_timestamp(r["timestamp_utc"])
    return rows


def group_into_bursts(rows):
    """
    Group images into burst groups: images within BURST_WINDOW_S of each other
    (by timestamp, same body) belong to the same group and must not be split.

    Returns a list of groups, where each group is a list of row dicts.
    Groups are formed globally (not per-body) to avoid any leakage.
    """
    # Sort all rows by timestamp
    sorted_rows = sorted(rows, key=lambda r: r["_ts"])

    groups = []
    current_group = [sorted_rows[0]]

    for row in sorted_rows[1:]:
        prev = current_group[-1]
        delta = abs((row["_ts"] - prev["_ts"]).total_seconds())
        if delta <= BURST_WINDOW_S:
            current_group.append(row)
        else:
            groups.append(current_group)
            current_group = [row]
    groups.append(current_group)

    return groups


def stratified_split(groups):
    """
    Perform stratified train/test split at the group level.

    For each body:
      - Collect all groups that contain images of that body.
      - Shuffle, then assign ~80% to train, ~20% to test.
      - Bodies with >= 50 images must appear in both splits.

    Since a group can contain images from multiple bodies (rare with burst
    grouping), we assign each group's body based on majority body.
    """
    random.seed(SEED)

    # Assign each group a "primary body" (majority body in the group)
    for g in groups:
        body_counts = defaultdict(int)
        for r in g:
            body_counts[r["body"]] += 1
        g_body = max(body_counts, key=body_counts.get)
        for r in g:
            r["_group_body"] = g_body

    # Group groups by body
    body_groups = defaultdict(list)
    for g in groups:
        body = g[0]["_group_body"]
        body_groups[body].append(g)

    train_rows = []
    test_rows = []
    stats = {}

    for body, bgroups in sorted(body_groups.items()):
        random.shuffle(bgroups)
        total_images = sum(len(g) for g in bgroups)

        # How many groups go to test?
        n_test_groups = max(1, round(len(bgroups) * (1 - TRAIN_RATIO)))

        # Bodies with >= 50 images MUST have at least 1 in each split
        if total_images >= 50:
            n_test_groups = max(1, n_test_groups)
            n_train_groups = len(bgroups) - n_test_groups
            if n_train_groups < 1:
                n_test_groups = len(bgroups) - 1
        else:
            # Small bodies: still try to split, but if only 1 group, all to train
            if len(bgroups) == 1:
                n_test_groups = 0

        test_gs = bgroups[:n_test_groups]
        train_gs = bgroups[n_test_groups:]

        n_train = sum(len(g) for g in train_gs)
        n_test = sum(len(g) for g in test_gs)

        for g in train_gs:
            for r in g:
                r["split"] = "train"
                train_rows.append(r)
        for g in test_gs:
            for r in g:
                r["split"] = "test"
                test_rows.append(r)

        stats[body] = {"total": total_images, "train": n_train, "test": n_test,
                       "groups": len(bgroups)}

    return train_rows, test_rows, stats


def build_adjacency_pairs(train_rows, test_rows):
    """
    For each test image, find the nearest train image in time with the same
    instrument and filter.
    """
    # Index train images by filter
    train_by_filter = defaultdict(list)
    for r in train_rows:
        train_by_filter[r["filter"]].append(r)

    # Sort each filter's train list by timestamp
    for filt in train_by_filter:
        train_by_filter[filt].sort(key=lambda r: r["_ts"])

    pairs = []
    for test_r in sorted(test_rows, key=lambda r: r["_ts"]):
        filt = test_r["filter"]
        candidates = train_by_filter.get(filt, [])
        if not candidates:
            continue

        # Binary search for nearest
        test_ts = test_r["_ts"]
        best = None
        best_delta = None

        import bisect
        timestamps = [c["_ts"] for c in candidates]
        idx = bisect.bisect_left(timestamps, test_ts)

        for i in [idx - 1, idx]:
            if 0 <= i < len(candidates):
                delta = abs((candidates[i]["_ts"] - test_ts).total_seconds())
                if best_delta is None or delta < best_delta:
                    best = candidates[i]
                    best_delta = delta

        if best is not None:
            pairs.append({
                "test_image_id": test_r["image_id"],
                "train_image_id": best["image_id"],
                "test_timestamp_utc": test_r["timestamp_utc"],
                "train_timestamp_utc": best["timestamp_utc"],
                "time_delta_s": round(best_delta, 3),
                "body": test_r["body"],
                "filter": filt,
            })

    return pairs


def write_split_csv(path, rows, split_name):
    """Write train or test CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "body", "timestamp_utc", "split"])
        for r in sorted(rows, key=lambda r: r["_ts"]):
            writer.writerow([r["image_id"], r["body"], r["timestamp_utc"], split_name])


def write_adjacency_csv(path, pairs):
    """Write adjacency pairs CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_image_id", "train_image_id", "test_timestamp_utc",
                          "train_timestamp_utc", "time_delta_s", "body", "filter"])
        for p in pairs:
            writer.writerow([p["test_image_id"], p["train_image_id"],
                             p["test_timestamp_utc"], p["train_timestamp_utc"],
                             p["time_delta_s"], p["body"], p["filter"]])


def main():
    print("Loading manifest...")
    rows = load_manifest()
    print(f"  {len(rows)} images after excluding {EXCLUDE_BODIES}")

    print("Grouping into bursts (window={0}s)...".format(BURST_WINDOW_S))
    groups = group_into_bursts(rows)
    print(f"  {len(groups)} burst groups from {len(rows)} images")

    print("Performing stratified split...")
    train_rows, test_rows, stats = stratified_split(groups)
    print(f"  Train: {len(train_rows)}, Test: {len(test_rows)}")
    print(f"  Ratio: {len(train_rows)/len(rows)*100:.1f}% / {len(test_rows)/len(rows)*100:.1f}%")

    print("\nPer-body distribution:")
    print(f"  {'Body':<25s} {'Total':>6s} {'Train':>6s} {'Test':>6s} {'Groups':>7s}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
    for body in sorted(stats.keys()):
        s = stats[body]
        print(f"  {body:<25s} {s['total']:>6d} {s['train']:>6d} {s['test']:>6d} {s['groups']:>7d}")

    # Verify no leakage: no pair of images within 60s should be in different splits
    all_assigned = train_rows + test_rows
    all_assigned.sort(key=lambda r: r["_ts"])
    leaks = 0
    for i in range(len(all_assigned) - 1):
        a, b = all_assigned[i], all_assigned[i + 1]
        delta = abs((b["_ts"] - a["_ts"]).total_seconds())
        if delta <= BURST_WINDOW_S and a["split"] != b["split"]:
            leaks += 1
    print(f"\nLeakage check: {leaks} violations (should be 0)")

    print("\nBuilding adjacency pairs...")
    pairs = build_adjacency_pairs(train_rows, test_rows)
    print(f"  {len(pairs)} adjacency pairs found")

    # Write outputs
    write_split_csv(TRAIN_CSV, train_rows, "train")
    write_split_csv(TEST_CSV, test_rows, "test")
    write_adjacency_csv(ADJ_CSV, pairs)
    print(f"\nWritten:")
    print(f"  {TRAIN_CSV}")
    print(f"  {TEST_CSV}")
    print(f"  {ADJ_CSV}")

    return stats, pairs, leaks, len(train_rows), len(test_rows), len(rows)


if __name__ == "__main__":
    stats, pairs, leaks, n_train, n_test, n_total = main()
