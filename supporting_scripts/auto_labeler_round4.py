#!/usr/bin/env python3
"""
Expansion auto-labeler for Cassini ISSNA imagery (Round 4).

Labels new images from the second expansion set while PRESERVING existing
approved labels (272 train + 53 val = 325). Same sanity-check thresholds
as previous rounds.

CRITICAL: All co-iss-w* images are excluded (ISSWA). Only co-iss-n* images.

Output:
- New YOLO label files in data/cassini_issna_yolo/labels/train/<image_id>.txt
- New images copied into data/cassini_issna_yolo/images/train/
- Review bundle in review/cassini_issna/auto_labels_round4/
- Stats JSON at artifacts/cassini_issna/auto_label_round4_stats.json
"""

import csv
import cv2
import json
import math
import os
import random
import sys
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
WORKSPACE = Path("/Users/upadhyay/dev/ICES/object_detection_opencv_cpp")
ARTIFACTS = WORKSPACE / "artifacts"
YOLO_DIR = WORKSPACE / "data" / "cassini_issna_yolo"
YOLO_LABELS_TRAIN = YOLO_DIR / "labels" / "train"
YOLO_IMAGES_TRAIN = YOLO_DIR / "images" / "train"
DATA_DIR = WORKSPACE / "data" / "cassini" / "issna"  # body-organized source
REVIEW_DIR = WORKSPACE / "review" / "cassini_issna" / "auto_labels_round4"

# Instrument parameters (from instruments.csv, verified)
FOCAL_LENGTH_MM = 2003.44
PIXEL_PITCH_UM = 12.0
FOCAL_LENGTH_M = FOCAL_LENGTH_MM * 1e-3
PIXEL_PITCH_M = PIXEL_PITCH_UM * 1e-6
IMAGE_SIZE = 1024

IFOV_RAD = PIXEL_PITCH_M / FOCAL_LENGTH_M  # ~5.989e-6 rad/px

# Thresholds (same as rounds 2-3)
MIN_DIAMETER_PX = 15
EDGE_MARGIN_FRAC = 0.05
CONTOUR_PROX_DIAM_MULT = 1.5
CONTOUR_PROX_MIN_PX = 100
BBOX_PAD_FRAC = 0.15
WHOLE_FRAME_FRAC = 0.85
ANCHORED_FALLBACK_FRAC = 0.50
MAX_ASPECT_RATIO = 3.0
RINGS_MIN_EXTENT_PX = 100

# Strict whole-frame-bbox rule: skip if bbox > 95% of frame
STRICT_WHOLE_FRAME_FRAC = 0.95

# Body radii in km (from IAU / NAIF pck00011.tpc)
BODY_RADII_KM = {
    "saturn":        58232.0,
    "saturn_rings":  117580.0,
    "titan":         2574.7,
    "rhea":          763.8,
    "iapetus":       734.5,
    "dione":         561.4,
    "tethys":        531.1,
    "enceladus":     252.1,
    "mimas":         198.2,
    "hyperion":      135.0,
    "phoebe":        106.5,
    "janus":         89.5,
    "epimetheus":    58.1,
    "prometheus":    43.1,
    "pandora":       40.7,
    "helene":        17.6,
    "telesto":       12.4,
    "calypso":       10.7,
    "atlas":         15.1,
    "pan":           14.1,
    "methone":       1.6,
    "pallene":       2.5,
    "polydeuces":    1.3,
    "daphnis":       3.8,
    "anthe":         0.9,
    "aegaeon":       0.3,
    "kiviuq":        8.0,
    "siarnaq":       20.0,
    "paaliaq":       11.0,
    "ijiraq":        6.0,
    "albiorix":      16.0,
    "erriapus":      5.0,
    "tarvos":        7.5,
    "ymir":          9.0,
    "thrymr":        3.5,
    "skathi":        4.0,
    "mundilfari":    3.5,
    "narvi":         3.5,
    "suttungr":      3.5,
    "bestla":        3.5,
    "hyrrokkin":     4.0,
    "bebhionn":      3.0,
    "bergelmir":     3.0,
    "fornjot":       3.0,
    "hati":          3.0,
    "tarqeq":        3.5,
    "greip":         3.0,
    "skoll":         3.0,
    "loge":          3.0,
    "jarnsaxa":      3.0,
    "surtur":        3.0,
    "kari":          3.5,
    "s_2004_s_12":   2.5,
    "s_2004_s_13":   3.0,
    "jupiter":       69911.0,
    "io":            1821.6,
    "europa":        1560.8,
    "ganymede":      2631.2,
    "callisto":      2410.3,
    "himalia":       85.0,
    "earth":         6371.0,
    "moon":          1737.4,
    "venus":         6051.8,
    "pluto":         1188.3,
}


def load_class_names():
    """Load class names from the 64-class file."""
    names = []
    with open(ARTIFACTS / "cassini_issna" / "class_names.txt") as f:
        for line in f:
            s = line.strip()
            if s:
                names.append(s)
    return names


def load_spacecraft_state():
    state = {}
    with open(ARTIFACTS / "spacecraft_state.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state[row["image_id"]] = row
    return state


def load_manifest():
    manifest = {}
    with open(ARTIFACTS / "cassini_issna" / "image_manifest.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["image_id"]] = row
    return manifest


def load_id_set(path):
    ids = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["image_id"])
    return ids


def load_drops():
    drops = set()
    drop_path = ARTIFACTS / "cassini_issna" / "dropped_by_human.txt"
    if drop_path.exists():
        with open(drop_path) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    drops.add(s)
    return drops


def load_isswa_excluded():
    """Load ISSWA exclusion list (plain text, one ID per line)."""
    excluded = set()
    path = ARTIFACTS / "cassini_issna" / "isswa_excluded.txt"
    if path.exists():
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s and not s.startswith("#"):
                    excluded.add(s)
    return excluded


def compute_range_km(state_row):
    x = float(state_row["position_km_x"])
    y = float(state_row["position_km_y"])
    z = float(state_row["position_km_z"])
    return math.sqrt(x*x + y*y + z*z)


def predicted_diameter_px(body_radius_km, range_km):
    if range_km <= 0 or body_radius_km <= 0:
        return 0.0
    angular_diameter_rad = 2.0 * math.atan(body_radius_km / range_km)
    return angular_diameter_rad / IFOV_RAD


def find_body_contour(img_gray, search_center, search_radius_px, predicted_diam_px):
    h, w = img_gray.shape[:2]
    cx, cy = int(search_center[0]), int(search_center[1])
    sr = max(int(search_radius_px), 50)

    x1 = max(0, cx - sr)
    y1 = max(0, cy - sr)
    x2 = min(w, cx + sr)
    y2 = min(h, cy + sr)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, 0.0, "search_window_too_small"

    roi = img_gray[y1:y2, x1:x2]
    roi_max = roi.max()

    if roi_max < 10:
        return None, 0.0, "no_signal_in_search_window"

    best_contour = None
    best_score = 0.0

    for thresh_method in ["otsu", "percentile_95", "percentile_90", "fixed_30"]:
        if thresh_method == "otsu":
            if roi_max < 20:
                continue
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif thresh_method == "percentile_95":
            t = max(np.percentile(roi, 95), 15)
            _, binary = cv2.threshold(roi, t, 255, cv2.THRESH_BINARY)
        elif thresh_method == "percentile_90":
            t = max(np.percentile(roi, 90), 10)
            _, binary = cv2.threshold(roi, t, 255, cv2.THRESH_BINARY)
        elif thresh_method == "fixed_30":
            _, binary = cv2.threshold(roi, 30, 255, cv2.THRESH_BINARY)

        if predicted_diam_px > 20:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            cnt_cx = bx + bw / 2.0
            cnt_cy = by + bh / 2.0

            expected_cx = cx - x1
            expected_cy = cy - y1
            dist = math.sqrt((cnt_cx - expected_cx)**2 + (cnt_cy - expected_cy)**2)

            cnt_diam = math.sqrt(bw * bh)
            if predicted_diam_px > 0:
                size_ratio = cnt_diam / predicted_diam_px
                size_score = max(0, 1.0 - abs(math.log(max(size_ratio, 0.01))))
            else:
                size_score = min(cnt_diam / 50.0, 1.0)

            prox_score = max(0, 1.0 - dist / max(sr, 1))
            score = 0.4 * size_score + 0.6 * prox_score

            if score > best_score:
                best_score = score
                best_contour = (bx + x1, by + y1, bw, bh)

    if best_contour is None:
        return None, 0.0, "no_contour_found"

    return best_contour, best_score, None


def pad_bbox(bbox, pad_frac, img_w, img_h):
    x, y, w, h = bbox
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_w - x, w + 2 * pad_x)
    h = min(img_h - y, h + 2 * pad_y)
    return (x, y, w, h)


def bbox_to_yolo(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def aspect_ratio(w, h):
    if w <= 0 or h <= 0:
        return float("inf")
    mn = min(w, h)
    mx = max(w, h)
    return mx / mn


def render_review_image(img_path, bbox, body_name, confidence, output_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    x, y, w, h = bbox
    color = (0, 255, 0)
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    label = f"{body_name} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
    label_y = max(y - 5, th + 5)
    cv2.rectangle(img, (x, label_y - th - 5), (x + tw + 4, label_y + 3), (0, 0, 0), -1)
    cv2.putText(img, label, (x + 2, label_y), font, 0.5, color, 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), img)
    return True


def main():
    print("=" * 60)
    print("ICES Expansion Auto-Labeler: cassini / issna  (ROUND 4)")
    print("=" * 60)

    # Load data
    class_names = load_class_names()
    class_id_map = {name: i for i, name in enumerate(class_names)}
    print(f"Classes: {len(class_names)}")

    manifest = load_manifest()
    spacecraft_state = load_spacecraft_state()
    train_ids = load_id_set(ARTIFACTS / "cassini_issna" / "train_split.csv")
    test_ids = load_id_set(ARTIFACTS / "cassini_issna" / "test_split.csv")
    dropped = load_drops()
    isswa_excluded = load_isswa_excluded()

    print(f"Manifest: {len(manifest)} images")
    print(f"Train split: {len(train_ids)} IDs")
    print(f"Test split: {len(test_ids)} IDs")
    print(f"Dropped: {len(dropped)} IDs")
    print(f"ISSWA excluded: {len(isswa_excluded)} IDs")
    print(f"Spacecraft state: {len(spacecraft_state)} rows")

    # Identify existing approved labels (DO NOT overwrite)
    existing_labels = set()
    for subdir in ["train", "val"]:
        label_dir = YOLO_DIR / "labels" / subdir
        if label_dir.exists():
            for f in label_dir.iterdir():
                if f.suffix == ".txt":
                    existing_labels.add(f.stem)
    print(f"Existing approved labels (preserved): {len(existing_labels)}")

    # All manifest IDs that are co-iss-n* only
    issna_manifest_ids = set()
    for mid in manifest:
        if mid.startswith("co-iss-n"):
            issna_manifest_ids.add(mid)
    print(f"ISSNA-only manifest IDs (co-iss-n*): {len(issna_manifest_ids)}")

    # Eligible for labeling:
    #   - In manifest AND co-iss-n* prefix
    #   - NOT in dropped_by_human.txt
    #   - NOT in isswa_excluded.txt
    #   - NOT in test_split.csv
    #   - NOT already labeled
    eligible_ids = issna_manifest_ids - dropped - isswa_excluded - test_ids - existing_labels
    print(f"Eligible for new labeling: {len(eligible_ids)}")

    # Statistics
    stats = {
        "total_eligible": len(eligible_ids),
        "preserved_labels": len(existing_labels),
        "new_accepted": 0,
        "new_rejected": 0,
        "rejected_reasons": defaultdict(int),
        "per_body_new_accepted": defaultdict(int),
        "per_body_new_rejected": defaultdict(int),
    }

    labeled_images = []
    rejected_images = []

    # Ensure output directories exist
    YOLO_LABELS_TRAIN.mkdir(parents=True, exist_ok=True)
    YOLO_IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)

    eligible_sorted = sorted(eligible_ids)
    for i, image_id in enumerate(eligible_sorted):
        if i % 200 == 0:
            print(f"Processing {i}/{len(eligible_sorted)}...")

        # CRITICAL: Double-check this is a co-iss-n* image
        if not image_id.startswith("co-iss-n"):
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["not_issna_prefix"] += 1
            rejected_images.append({"image_id": image_id, "body": "?", "reason": "not_issna_prefix"})
            continue

        mrow = manifest.get(image_id)
        if mrow is None:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["not_in_manifest"] += 1
            rejected_images.append({"image_id": image_id, "body": "?", "reason": "not_in_manifest"})
            continue

        body = mrow["body"]

        # Skip unknown targets
        if body == "unknown" or body not in class_id_map:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["unknown_or_unlisted_body"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "unknown_or_unlisted_body"})
            continue

        # Check state_verified
        sv = mrow.get("state_verified", "")
        if sv.lower() == "false":
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["state_verified_false"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "state_verified_false"})
            continue

        # Find image file
        local_path = mrow.get("local_path", "")
        img_path = Path(local_path) if local_path else None

        # Also check body-organized dir
        if img_path is None or not img_path.exists():
            img_path = DATA_DIR / body / "images" / f"{image_id}.png"
        if not img_path.exists():
            # Check YOLO images dir
            img_path = YOLO_IMAGES_TRAIN / f"{image_id}.png"
        if not img_path.exists():
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["image_file_missing"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "image_file_missing"})
            continue

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["image_load_failed"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "image_load_failed"})
            continue

        img_h, img_w = img.shape[:2]

        # Body radius
        body_radius_km = BODY_RADII_KM.get(body, None)
        if body_radius_km is None:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["unknown_body_radius"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "unknown_body_radius"})
            continue

        # SPICE state required
        if image_id not in spacecraft_state:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["no_spice_state"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "no_spice_state"})
            continue

        state = spacecraft_state[image_id]
        try:
            range_km = compute_range_km(state)
        except (ValueError, KeyError):
            range_km = 0.0

        if range_km <= 0:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["invalid_spice_range"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "invalid_spice_range"})
            continue

        predicted_diam = predicted_diameter_px(body_radius_km, range_km)

        # Minimum diameter check
        min_req = RINGS_MIN_EXTENT_PX if body == "saturn_rings" else MIN_DIAMETER_PX
        if predicted_diam < min_req:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["predicted_too_small"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body,
                                    "reason": f"predicted_too_small ({predicted_diam:.1f} px, min {min_req})"})
            continue

        # Body subtends > 95% of frame -> skip
        if predicted_diam > STRICT_WHOLE_FRAME_FRAC * max(img_w, img_h):
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["predicted_exceeds_95pct_frame"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body,
                                    "reason": f"predicted_exceeds_95pct_frame ({predicted_diam:.0f} px)"})
            continue

        # Predicted center = image center (no per-image CK projection)
        predicted_center = (img_w / 2.0, img_h / 2.0)
        px_cx, px_cy = predicted_center

        # Edge margin check
        if (px_cx < img_w * EDGE_MARGIN_FRAC or
            px_cx > img_w * (1 - EDGE_MARGIN_FRAC) or
            px_cy < img_h * EDGE_MARGIN_FRAC or
            px_cy > img_h * (1 - EDGE_MARGIN_FRAC)):
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["predicted_center_off_frame"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "predicted_center_off_frame"})
            continue

        # Contour search
        search_radius = max(predicted_diam * 2, 100)
        bbox, confidence, rejection = find_body_contour(img, predicted_center, search_radius, predicted_diam)

        if bbox is None:
            stats["new_rejected"] += 1
            reason = rejection or "no_contour_found"
            stats["rejected_reasons"][reason] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": reason})
            continue

        bx, by, bw, bh = bbox

        # Contour proximity check
        bcx = bx + bw / 2.0
        bcy = by + bh / 2.0
        dist = math.sqrt((bcx - px_cx)**2 + (bcy - px_cy)**2)
        prox_limit = max(predicted_diam * CONTOUR_PROX_DIAM_MULT, CONTOUR_PROX_MIN_PX)
        if dist > prox_limit:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["contour_far_from_predicted"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body,
                                    "reason": f"contour_far_from_predicted ({dist:.0f}px > {prox_limit:.0f}px)"})
            continue

        # Whole-frame bbox check
        if bw > WHOLE_FRAME_FRAC * img_w and bh > WHOLE_FRAME_FRAC * img_h:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["whole_frame_bbox"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "whole_frame_bbox"})
            continue

        # Anchored fallback check
        if (bx == 0 and by == 0 and
            (bw > ANCHORED_FALLBACK_FRAC * img_w or bh > ANCHORED_FALLBACK_FRAC * img_h)):
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["anchored_fallback_bbox"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body, "reason": "anchored_fallback_bbox"})
            continue

        # Aspect ratio check
        ar = aspect_ratio(bw, bh)
        if ar > MAX_ASPECT_RATIO:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["extreme_aspect_ratio"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body,
                                    "reason": f"extreme_aspect_ratio ({ar:.1f})"})
            continue

        # STRICT: reject if padded bbox > 95% of frame width or height
        padded = pad_bbox((bx, by, bw, bh), BBOX_PAD_FRAC, img_w, img_h)
        px, py, pw, ph = padded
        if pw > STRICT_WHOLE_FRAME_FRAC * img_w or ph > STRICT_WHOLE_FRAME_FRAC * img_h:
            stats["new_rejected"] += 1
            stats["rejected_reasons"]["padded_bbox_exceeds_95pct"] += 1
            stats["per_body_new_rejected"][body] += 1
            rejected_images.append({"image_id": image_id, "body": body,
                                    "reason": f"padded_bbox_exceeds_95pct ({pw}x{ph})"})
            continue

        # Near-edge confidence softening
        if (bcx < img_w * EDGE_MARGIN_FRAC or
            bcx > img_w * (1 - EDGE_MARGIN_FRAC) or
            bcy < img_h * EDGE_MARGIN_FRAC or
            bcy > img_h * (1 - EDGE_MARGIN_FRAC)):
            confidence *= 0.8

        bbox_final = padded
        class_id = class_id_map[body]
        cx_norm, cy_norm, w_norm, h_norm = bbox_to_yolo(bbox_final, img_w, img_h)

        # Write YOLO label
        label_path = YOLO_LABELS_TRAIN / f"{image_id}.txt"
        with open(label_path, "w") as f:
            f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        # Ensure image is in YOLO images dir
        yolo_img_path = YOLO_IMAGES_TRAIN / f"{image_id}.png"
        if not yolo_img_path.exists():
            shutil.copy2(str(img_path), str(yolo_img_path))

        stats["new_accepted"] += 1
        stats["per_body_new_accepted"][body] += 1

        labeled_images.append({
            "image_id": image_id,
            "body": body,
            "img_path": str(img_path),
            "label_path": str(label_path),
            "bbox": bbox_final,
            "yolo": (cx_norm, cy_norm, w_norm, h_norm),
            "confidence": confidence,
            "predicted_diam_px": predicted_diam,
            "class_id": class_id,
            "is_new": image_id not in train_ids,
        })

    total_labels = stats["new_accepted"] + len(existing_labels)
    print(f"\n{'='*60}")
    print(f"Round 4 expansion labeling complete:")
    print(f"  New labels generated: {stats['new_accepted']}")
    print(f"  Preserved existing labels: {len(existing_labels)}")
    print(f"  Total training labels: {total_labels}")
    print(f"  New rejected: {stats['new_rejected']}")
    print(f"\nRejection reasons:")
    for reason, count in sorted(stats["rejected_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Per-body stats for combined set
    existing_body_counts = defaultdict(int)
    for lbl_id in existing_labels:
        lbl_file = None
        for subdir in ["train", "val"]:
            p = YOLO_DIR / "labels" / subdir / f"{lbl_id}.txt"
            if p.exists():
                lbl_file = p
                break
        if lbl_file:
            with open(lbl_file) as f:
                line = f.readline().strip()
                if line:
                    cid = int(line.split()[0])
                    if 0 <= cid < len(class_names):
                        existing_body_counts[class_names[cid]] += 1

    combined_body_counts = defaultdict(int)
    for k, v in existing_body_counts.items():
        combined_body_counts[k] += v
    for k, v in stats["per_body_new_accepted"].items():
        combined_body_counts[k] += v

    print(f"\nPer-class label counts (combined):")
    for body_name in sorted(combined_body_counts.keys()):
        cid = class_id_map.get(body_name, -1)
        cnt = combined_body_counts[body_name]
        print(f"  {cid:3d} {body_name:20s}: {cnt}")

    # Write stats JSON
    stats_out = {
        "round": 4,
        "type": "expansion_round4",
        "total_eligible": stats["total_eligible"],
        "preserved_labels": len(existing_labels),
        "new_accepted": stats["new_accepted"],
        "new_rejected": stats["new_rejected"],
        "total_labels": total_labels,
        "rejected_reasons": dict(stats["rejected_reasons"]),
        "per_body_new_accepted": dict(stats["per_body_new_accepted"]),
        "per_body_new_rejected": dict(stats["per_body_new_rejected"]),
        "combined_body_counts": dict(combined_body_counts),
        "class_names": class_names,
    }
    stats_path = ARTIFACTS / "cassini_issna" / "auto_label_round4_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"\nWrote {stats_path}")

    # ============================================================
    # Generate human review bundle (25 per bucket per brief 8.5)
    # ============================================================
    print(f"\n{'='*60}")
    print("Generating human review bundle (round 4 expansion)...")

    if len(labeled_images) == 0:
        print("WARNING: No new images were labeled. Review bundle will be empty.")

    # Categorize
    easy = []
    hard = []
    ambiguous = []

    for item in labeled_images:
        diam = item["predicted_diam_px"]
        conf = item["confidence"]
        bbox = item["bbox"]
        img_cx = bbox[0] + bbox[2] / 2.0
        img_cy = bbox[1] + bbox[3] / 2.0
        dist_from_center = math.sqrt(
            (img_cx - IMAGE_SIZE / 2)**2 + (img_cy - IMAGE_SIZE / 2)**2
        )
        near_center = dist_from_center < 0.4 * (IMAGE_SIZE / 2)

        if conf < 0.5:
            ambiguous.append(item)
        elif diam > 50 and near_center and conf >= 0.7:
            easy.append(item)
        elif diam < 30 or not near_center or conf < 0.7:
            hard.append(item)
        else:
            easy.append(item)

    print(f"Categorized: {len(easy)} easy, {len(hard)} hard, {len(ambiguous)} ambiguous")

    # Fresh seed for round 4
    random.seed(20260410_04)

    def sample_up_to(lst, n):
        if len(lst) <= n:
            return lst[:]
        return random.sample(lst, n)

    easy_sample = sample_up_to(easy, 25)
    hard_sample = sample_up_to(hard, 25)
    ambiguous_sample = sample_up_to(ambiguous, 25)

    # Random sample from non-overlapping pool
    sampled_ids = set(item["image_id"] for item in easy_sample + hard_sample + ambiguous_sample)
    random_pool = [item for item in labeled_images if item["image_id"] not in sampled_ids]
    random_sample = sample_up_to(random_pool, 25) if random_pool else []

    # Create review directories
    for bucket_name in ["easy", "hard", "ambiguous", "random"]:
        bucket_dir = REVIEW_DIR / bucket_name
        bucket_dir.mkdir(parents=True, exist_ok=True)
        for old in bucket_dir.glob("*.png"):
            old.unlink()

    # Render overlay images
    index_rows = []
    for bucket_name, bucket_sample in [
        ("easy", easy_sample),
        ("hard", hard_sample),
        ("ambiguous", ambiguous_sample),
        ("random", random_sample),
    ]:
        for item in bucket_sample:
            img_path = Path(item["img_path"])
            output_path = REVIEW_DIR / bucket_name / f"{item['image_id']}.png"

            success = render_review_image(
                img_path, item["bbox"], item["body"],
                item["confidence"], output_path
            )

            if success:
                index_rows.append({
                    "bucket": bucket_name,
                    "filename": f"{item['image_id']}.png",
                    "body": item["body"],
                    "bbox_x": item["bbox"][0],
                    "bbox_y": item["bbox"][1],
                    "bbox_w": item["bbox"][2],
                    "bbox_h": item["bbox"][3],
                    "confidence": f"{item['confidence']:.3f}",
                    "predicted_diam_px": f"{item['predicted_diam_px']:.1f}",
                    "image_id": item["image_id"],
                    "is_new": item["is_new"],
                })

    # Write INDEX.md
    index_path = REVIEW_DIR / "INDEX.md"
    with open(index_path, "w") as f:
        f.write("# Auto-Label Review Index: cassini / issna  (ROUND 4 -- SECOND EXPANSION)\n\n")
        f.write(f"Generated: 2026-04-10\n\n")
        f.write(f"New labels: {stats['new_accepted']} | Preserved: {len(existing_labels)} | Total: {total_labels}\n\n")
        f.write("## Sample (round-4 seed)\n\n")
        f.write("| Bucket | Filename | Body | BBox (x,y,w,h) | Confidence | Pred Diam (px) | New? | Image ID |\n")
        f.write("|--------|----------|------|-----------------|------------|----------------|------|----------|\n")
        for r in index_rows:
            f.write(f"| {r['bucket']} | {r['filename']} | {r['body']} | "
                    f"({r['bbox_x']},{r['bbox_y']},{r['bbox_w']},{r['bbox_h']}) | "
                    f"{r['confidence']} | {r['predicted_diam_px']} | "
                    f"{'YES' if r['is_new'] else 'no'} | "
                    f"{r['image_id']} |\n")

    print(f"Wrote {index_path}")

    # Write DECISION.md (empty -- human fills it in)
    decision_path = REVIEW_DIR / "DECISION.md"
    with open(decision_path, "w") as f:
        f.write("# Decision: Auto-Labels Round 4 (Second Expansion) -- cassini / issna\n\n")
        f.write("STATUS: \n\n")
        f.write("## Per-image verdicts (optional)\n\n")
        f.write("<!-- Mark individual images if needed:\n")
        f.write("image_id | verdict (ok / x) | notes\n")
        f.write("-->\n\n")
        f.write("## Instructions for agent (if NEEDS_FIXES)\n\n")
        f.write("<!-- Describe what to fix -->\n")
    print(f"Wrote {decision_path}")

    # Write REVIEW_REQUEST.md
    review_request_path = REVIEW_DIR / "REVIEW_REQUEST.md"
    with open(review_request_path, "w") as f:
        f.write("# Review Request: Auto-Labels for cassini / issna  (ROUND 4 -- SECOND EXPANSION)\n\n")
        f.write("This is the second expansion labeling round (round 4). The 325 labels from\n")
        f.write("previous rounds (272 train + 53 val) are preserved. New labels were generated\n")
        f.write("for previously-unlabeled co-iss-n* images using the same thresholds as rounds 2-3.\n\n")
        f.write("CRITICAL: All co-iss-w* (ISSWA) images were excluded from labeling.\n\n")
        f.write("**Iteration log:** `research_log/0017_auto_label_round4.md`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **New labels generated (this round):** {stats['new_accepted']}\n")
        f.write(f"- **Preserved existing labels:** {len(existing_labels)}\n")
        f.write(f"- **Total training labels available:** {total_labels}\n")
        f.write(f"- **New images rejected:** {stats['new_rejected']}\n")
        rej_rate = stats['new_rejected'] / max(stats['total_eligible'], 1) * 100
        f.write(f"- **Rejection rate (new images only):** {rej_rate:.1f}%\n\n")

        f.write("## Rejection Reasons (new images only)\n\n")
        f.write("| Reason | Count |\n")
        f.write("|--------|-------|\n")
        for reason, count in sorted(stats["rejected_reasons"].items(), key=lambda x: -x[1]):
            f.write(f"| {reason} | {count} |\n")

        f.write("\n## Per-Body Statistics (new labels only)\n\n")
        f.write("| Body | New Accepted | New Rejected |\n")
        f.write("|------|-------------|-------------|\n")
        all_bodies = sorted(set(list(stats["per_body_new_accepted"].keys()) +
                               list(stats["per_body_new_rejected"].keys())))
        for body_name in all_bodies:
            acc = stats["per_body_new_accepted"].get(body_name, 0)
            rej = stats["per_body_new_rejected"].get(body_name, 0)
            f.write(f"| {body_name} | {acc} | {rej} |\n")

        f.write("\n## Combined Per-Class Counts\n\n")
        f.write("| Class ID | Body | Count |\n")
        f.write("|----------|------|-------|\n")
        for body_name in sorted(combined_body_counts.keys()):
            cid = class_id_map.get(body_name, -1)
            cnt = combined_body_counts[body_name]
            f.write(f"| {cid} | {body_name} | {cnt} |\n")

        f.write("\n## Review Buckets\n\n")
        f.write(f"Four buckets of sample images (25 each) in `review/cassini_issna/auto_labels_round4/`:\n\n")
        f.write(f"1. **easy/** ({len(easy_sample)} images) -- Large, well-centered, high confidence.\n")
        f.write(f"   These should be near-perfect; if any are wrong, the labeler is broken.\n")
        f.write(f"2. **hard/** ({len(hard_sample)} images) -- Small (<30px) or off-center.\n")
        f.write(f"   Most failures will live here.\n")
        f.write(f"3. **ambiguous/** ({len(ambiguous_sample)} images) -- Low confidence (<0.5).\n")
        f.write(f"   Judgment calls; co-visible bodies or faint targets.\n")
        f.write(f"4. **random/** ({len(random_sample)} images) -- Uniform random, catches categorization bias.\n")
        f.write(f"   If easy/hard/ambiguous look fine but random is garbage, the categorization is biased.\n\n")

        f.write("## What To Look For\n\n")
        f.write("- Does the green bounding box tightly enclose the correct body?\n")
        f.write("- Are expansion images (marked 'YES' in the New? column) labeled as well as the originals?\n")
        f.write("- Any systematic patterns in the rejections that suggest a bug?\n")
        f.write("- No ISSWA (co-iss-w*) images should appear anywhere in the labeled set.\n\n")

        f.write("## What To Do\n\n")
        f.write("Open each bucket folder and inspect the overlay PNGs.\n")
        f.write("Then write your verdict into:\n")
        f.write("`review/cassini_issna/auto_labels_round4/DECISION.md`\n\n")
        f.write("Write one of:\n")
        f.write("- `STATUS: APPROVED`\n")
        f.write("- `STATUS: NEEDS_FIXES` (with instructions)\n")
        f.write("- `STATUS: REJECTED`\n\n")

        f.write("## The Question\n\n")
        f.write("What would make you trust this labeler enough to start training?\n")

    print(f"Wrote {review_request_path}")
    print(f"\nReview bundle complete. {len(index_rows)} overlay images rendered.")

    # Return summary for the orchestrator
    print(f"\n{'='*60}")
    print("SUMMARY FOR ORCHESTRATOR:")
    print(f"  new_labels={stats['new_accepted']}")
    print(f"  preserved_labels={len(existing_labels)}")
    print(f"  total_labels={total_labels}")
    print(f"  new_rejected={stats['new_rejected']}")
    print(f"  review_bundle_ready=True")
    print(f"  review_request_path={review_request_path}")
    print("DONE.")

    return stats_out


if __name__ == "__main__":
    main()
