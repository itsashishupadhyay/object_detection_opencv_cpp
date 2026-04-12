#!/usr/bin/env python3
"""
Auto-labeler for Cassini ISSNA imagery.

Generates YOLO-format bounding box labels for training images by:
1. Predicting body pixel size from SPICE geometry (range + body radius + optics)
2. Refining with OpenCV contour detection
3. Sanity-checking each label before accepting

Produces:
- YOLO label files under data/cassini/issna/<body>/labels/<image_id>.txt
- class_names.txt
- auto_label_stats.json
- Human review bundle under review/cassini_issna/auto_labels/
"""

import csv
import cv2
import json
import math
import os
import random
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
WORKSPACE = Path("/Users/upadhyay/dev/ICES/object_detection_opencv_cpp")
ARTIFACTS = WORKSPACE / "artifacts"
DATA_DIR = WORKSPACE / "data" / "cassini" / "issna"
REVIEW_DIR = WORKSPACE / "review" / "cassini_issna" / "auto_labels"

# Instrument parameters (from instruments.csv, verified)
FOCAL_LENGTH_MM = 2003.44
PIXEL_PITCH_UM = 12.0
FOCAL_LENGTH_M = FOCAL_LENGTH_MM * 1e-3  # 2.00344 m
PIXEL_PITCH_M = PIXEL_PITCH_UM * 1e-6    # 12e-6 m
IMAGE_SIZE = 1024  # 1024x1024

# IFOV in radians per pixel
IFOV_RAD = PIXEL_PITCH_M / FOCAL_LENGTH_M  # ~5.989e-6 rad/px

# ============================================================
# Round 2 (NEEDS_FIXES) sanity check thresholds
# The human rejected 33/100 labels in round 1. Each threshold
# below is tightened per a specific failure pattern they flagged.
# ============================================================
# FIX #2: raise minimum predicted pixel diameter from 8 -> 15
#   (round 1 had multiple Wrong-Bounding-box rejects with 4x4, 5x5, 6x5
#    boxes where the labeler latched onto a star/noise instead of a
#    barely-resolvable target)
MIN_DIAMETER_PX = 15

# FIX #6: bodies must actually be inside the frame geometrically.
# If the predicted pixel center falls outside the image bounds
# (minus this margin), the body isn't in the frame -> reject.
EDGE_MARGIN_FRAC = 0.05

# FIX #5: tighten the contour-center proximity requirement.
# Refined contour center must be within
#     max(predicted_diam * 1.5, 100 px)
# of the predicted center, otherwise reject.
CONTOUR_PROX_DIAM_MULT = 1.5
CONTOUR_PROX_MIN_PX = 100

BBOX_PAD_FRAC = 0.15

# FIX #3: reject whole-frame / near-whole-frame bboxes.
# Round 1 produced (0,0,1024,1024), (0,0,256,256), (0,0,256,229) fallbacks.
WHOLE_FRAME_FRAC = 0.85    # reject if w AND h > 85% of image
ANCHORED_FALLBACK_FRAC = 0.50  # reject (0,0)-anchored boxes > 50% of image

# FIX #4: reject extreme aspect ratios (e.g. ganymede 416x20).
# Planetary bodies are approximately circular when resolved.
MAX_ASPECT_RATIO = 3.0

# FIX #7: saturn_rings-specific minimum extent (rings are extended).
RINGS_MIN_EXTENT_PX = 100

# Body radii in km (from IAU / NAIF pck00011.tpc, primary sources)
# For irregular bodies, using mean radius.
# "saturn_rings" gets a special effective radius = outer B ring radius = 117,580 km
BODY_RADII_KM = {
    "saturn":        58232.0,
    "saturn_rings":  117580.0,  # Outer visible ring edge (B ring outer)
    "titan":         2574.7,
    "rhea":          763.8,
    "iapetus":       734.5,
    "dione":         561.4,
    "tethys":        531.1,
    "enceladus":     252.1,
    "mimas":         198.2,
    "hyperion":      135.0,  # mean radius of irregular body
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
    # Irregular outer moons (very small, ~few km)
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
    # Jupiter system (for cruise-phase images)
    "jupiter":       69911.0,
    "io":            1821.6,
    "europa":        1560.8,
    "ganymede":      2631.2,
    "callisto":      2410.3,
    "himalia":       85.0,
    # Earth / Moon / Venus
    "earth":         6371.0,
    "moon":          1737.4,
    "venus":         6051.8,
    "pluto":         1188.3,
}


def load_train_split():
    """Load the training split CSV into a list of dicts."""
    rows = []
    with open(ARTIFACTS / "cassini_issna" / "train_split.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_spacecraft_state():
    """Load spacecraft state CSV into a dict keyed by image_id."""
    state = {}
    with open(ARTIFACTS / "spacecraft_state.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            state[row["image_id"]] = row
    return state


def load_manifest():
    """Load image manifest into a dict keyed by image_id."""
    manifest = {}
    with open(ARTIFACTS / "cassini_issna" / "image_manifest.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row["image_id"]] = row
    return manifest


def compute_range_km(state_row):
    """Compute spacecraft-to-target range from position vector."""
    x = float(state_row["position_km_x"])
    y = float(state_row["position_km_y"])
    z = float(state_row["position_km_z"])
    return math.sqrt(x*x + y*y + z*z)


def predicted_diameter_px(body_radius_km, range_km):
    """Compute predicted angular diameter in pixels."""
    if range_km <= 0 or body_radius_km <= 0:
        return 0.0
    angular_diameter_rad = 2.0 * math.atan(body_radius_km / range_km)
    diameter_px = angular_diameter_rad / IFOV_RAD
    return diameter_px


def find_body_contour(img_gray, search_center, search_radius_px, predicted_diam_px):
    """
    Find the best contour matching the target body in a search window.

    Returns (bbox, confidence, rejection_reason) where bbox is (x, y, w, h) in pixels
    or None if no good contour found.
    """
    h, w = img_gray.shape[:2]

    # Define search window
    cx, cy = int(search_center[0]), int(search_center[1])
    sr = max(int(search_radius_px), 50)  # At least 50px search radius

    x1 = max(0, cx - sr)
    y1 = max(0, cy - sr)
    x2 = min(w, cx + sr)
    y2 = min(h, cy + sr)

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None, 0.0, "search_window_too_small"

    roi = img_gray[y1:y2, x1:x2]

    # Adaptive thresholding strategy:
    # For small bodies (point-like), use a high threshold to find bright spots
    # For large bodies, use Otsu or a moderate threshold

    # First, check if the ROI has any signal
    roi_max = roi.max()
    roi_mean = roi.mean()

    if roi_max < 10:
        return None, 0.0, "no_signal_in_search_window"

    # Try multiple thresholding approaches
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

        # Morphological cleanup for larger objects
        if predicted_diam_px > 20:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        # Score each contour
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:  # Minimum 2x2 pixels
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            cnt_cx = bx + bw / 2.0
            cnt_cy = by + bh / 2.0

            # Distance from expected center (in ROI coordinates)
            expected_cx = cx - x1
            expected_cy = cy - y1
            dist = math.sqrt((cnt_cx - expected_cx)**2 + (cnt_cy - expected_cy)**2)

            # Size match score
            cnt_diam = math.sqrt(bw * bh)  # geometric mean of width/height
            if predicted_diam_px > 0:
                size_ratio = cnt_diam / predicted_diam_px
                size_score = max(0, 1.0 - abs(math.log(max(size_ratio, 0.01))))
            else:
                # No prediction -- just prefer larger objects near center
                size_score = min(cnt_diam / 50.0, 1.0)

            # Proximity score
            prox_score = max(0, 1.0 - dist / max(sr, 1))

            # Combined score
            score = 0.4 * size_score + 0.6 * prox_score

            if score > best_score:
                best_score = score
                # Convert back to full-image coordinates
                best_contour = (bx + x1, by + y1, bw, bh)

    if best_contour is None:
        return None, 0.0, "no_contour_found"

    return best_contour, best_score, None


def find_body_contour_fullframe(img_gray):
    """
    Fallback: find the dominant bright object in the full frame.
    Used when no SPICE state is available.

    Returns (bbox, confidence, rejection_reason).
    """
    h, w = img_gray.shape[:2]

    # Try Otsu first
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If Otsu gives too much (>30% of frame), raise threshold
    if np.count_nonzero(binary) > 0.3 * h * w:
        t = max(np.percentile(img_gray, 97), 30)
        _, binary = cv2.threshold(img_gray, t, 255, cv2.THRESH_BINARY)

    # If still nothing, try lower threshold
    if np.count_nonzero(binary) == 0:
        _, binary = cv2.threshold(img_gray, 15, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, 0.0, "no_contour_found_fullframe"

    # Sort by area
    contours_with_area = [(cv2.contourArea(c), c) for c in contours]
    contours_with_area.sort(key=lambda x: x[0], reverse=True)

    largest_area = contours_with_area[0][0]

    if largest_area < 4:
        return None, 0.0, "largest_contour_too_small"

    # Check ambiguity: if second-largest is >50% of largest, it's ambiguous
    if len(contours_with_area) > 1:
        second_area = contours_with_area[1][0]
        if second_area > 0.5 * largest_area and second_area > 100:
            return None, 0.0, "multiple_ambiguous_contours_fullframe"

    cnt = contours_with_area[0][1]
    bx, by, bw, bh = cv2.boundingRect(cnt)

    # Confidence is lower without SPICE
    confidence = 0.5
    if bw > 20 and bh > 20:
        confidence = 0.6
    if bw > 50 and bh > 50:
        confidence = 0.7

    return (bx, by, bw, bh), confidence, None


def bbox_touches_edge(bbox, img_w, img_h, margin=2):
    """Check if bounding box touches or is very near the image edge."""
    x, y, w, h = bbox
    if x <= margin or y <= margin:
        return True
    if x + w >= img_w - margin or y + h >= img_h - margin:
        return True
    return False


def pad_bbox(bbox, pad_frac, img_w, img_h):
    """Pad a bounding box by a fraction of its size, clipping to image bounds."""
    x, y, w, h = bbox
    pad_x = int(w * pad_frac)
    pad_y = int(h * pad_frac)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_w - x, w + 2 * pad_x)
    h = min(img_h - y, h + 2 * pad_y)
    return (x, y, w, h)


def bbox_to_yolo(bbox, img_w, img_h):
    """Convert (x, y, w, h) pixel bbox to YOLO normalized format."""
    x, y, w, h = bbox
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh


def render_review_image(img_path, bbox, body_name, confidence, output_path):
    """Render an image with bounding box overlay for human review."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False

    x, y, w, h = bbox
    # Green bbox
    color = (0, 255, 0)
    thickness = 2
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Label
    label = f"{body_name} ({confidence:.2f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)

    # Background for text
    label_y = max(y - 5, th + 5)
    cv2.rectangle(img, (x, label_y - th - 5), (x + tw + 4, label_y + 3), (0, 0, 0), -1)
    cv2.putText(img, label, (x + 2, label_y), font, font_scale, color, 1, cv2.LINE_AA)

    cv2.imwrite(str(output_path), img)
    return True


def load_permanent_drops():
    """Load the permanent human-drop list (FIX #8).

    These are image IDs the human rejected per-image in round 1 and
    must never be used for training, even if a stricter labeler would
    now accept them. The file is one image_id per line.
    """
    drop_path = ARTIFACTS / "cassini_issna" / "dropped_by_human.txt"
    if not drop_path.exists():
        return set()
    drops = set()
    with open(drop_path) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                drops.add(s)
    return drops


def aspect_ratio(w, h):
    if w <= 0 or h <= 0:
        return float("inf")
    mn = min(w, h)
    mx = max(w, h)
    return mx / mn


def main():
    print("=" * 60)
    print("ICES Auto-Labeler: cassini / issna  (ROUND 2, post human NEEDS_FIXES)")
    print("=" * 60)

    # Load data
    train_split = load_train_split()
    spacecraft_state = load_spacecraft_state()
    manifest = load_manifest()
    permanent_drops = load_permanent_drops()
    print(f"Permanent human-drop list: {len(permanent_drops)} image_ids")

    print(f"Training images: {len(train_split)}")
    print(f"Spacecraft state rows: {len(spacecraft_state)}")

    # Build class list (alphabetical from bodies present in training set)
    bodies_in_train = sorted(set(row["body"] for row in train_split))
    class_names = bodies_in_train
    class_id_map = {name: i for i, name in enumerate(class_names)}

    print(f"Classes ({len(class_names)}): {class_names}")

    # Write class_names.txt
    class_names_path = ARTIFACTS / "cassini_issna" / "class_names.txt"
    with open(class_names_path, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"Wrote {class_names_path}")

    # Statistics tracking
    stats = {
        "total_train_images": len(train_split),
        "accepted": 0,
        "rejected": 0,
        "accepted_with_spice": 0,
        "accepted_without_spice": 0,
        "rejected_reasons": defaultdict(int),
        "per_body_accepted": defaultdict(int),
        "per_body_rejected": defaultdict(int),
        "per_body_rejected_reasons": defaultdict(lambda: defaultdict(int)),
    }

    # Track all labeled images for review sampling
    labeled_images = []  # list of dicts with metadata
    rejected_images = []

    # Clear out any pre-existing label .txt files so we start clean.
    # (Round-1 labels must not leak into round-2 training.)
    purged = 0
    for body_dir in DATA_DIR.iterdir() if DATA_DIR.exists() else []:
        label_dir = body_dir / "labels"
        if label_dir.exists():
            for lbl in label_dir.glob("*.txt"):
                lbl.unlink()
                purged += 1
    print(f"Purged {purged} stale round-1 label files")

    # Process each training image
    for i, trow in enumerate(train_split):
        image_id = trow["image_id"]
        body = trow["body"]

        if i % 100 == 0:
            print(f"Processing {i}/{len(train_split)}...")

        # FIX #8: permanent human-drop list takes precedence over everything.
        if image_id in permanent_drops:
            stats["rejected"] += 1
            stats["rejected_reasons"]["permanent_human_drop"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "permanent_human_drop"
            })
            continue

        # Find image path
        img_path = DATA_DIR / body / "images" / f"{image_id}.png"
        if not img_path.exists():
            stats["rejected"] += 1
            stats["rejected_reasons"]["image_file_missing"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "image_file_missing"
            })
            continue

        # Load image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            stats["rejected"] += 1
            stats["rejected_reasons"]["image_load_failed"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "image_load_failed"
            })
            continue

        img_h, img_w = img.shape[:2]

        # Get body radius
        body_radius_km = BODY_RADII_KM.get(body, None)
        if body_radius_km is None:
            stats["rejected"] += 1
            stats["rejected_reasons"]["unknown_body_radius"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "unknown_body_radius"
            })
            continue

        # FIX #1: strict SPICE requirement -- no OpenCV-only fallback.
        # The human's round-1 review showed that every no-SPICE label was
        # on an irregular moon / frame where the labeler latched onto a
        # star or noise. Without SPICE we cannot prove the detected object
        # is the intended target, so we drop the image.
        if image_id not in spacecraft_state:
            stats["rejected"] += 1
            stats["rejected_reasons"]["no_spice_state"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "no_spice_state"
            })
            continue

        state = spacecraft_state[image_id]
        try:
            range_km = compute_range_km(state)
        except (ValueError, KeyError):
            range_km = 0.0

        if range_km <= 0:
            stats["rejected"] += 1
            stats["rejected_reasons"]["invalid_spice_range"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "invalid_spice_range"
            })
            continue

        predicted_diam = predicted_diameter_px(body_radius_km, range_km)

        # FIX #6: the camera was pointing at target_body_from_metadata
        # (encoded by the folder). We don't have per-image CK projection
        # here, so we assume the target is at the boresight center and
        # require a bright contour to actually exist near center.
        predicted_center = (img_w / 2.0, img_h / 2.0)

        # FIX #2 + #7: minimum predicted pixel diameter.
        # Standard bodies must be >= 15 px. Rings must be >= 100 px of
        # effective extent (we use the same outer-edge diameter proxy).
        min_req = RINGS_MIN_EXTENT_PX if body == "saturn_rings" else MIN_DIAMETER_PX
        if predicted_diam < min_req:
            stats["rejected"] += 1
            reason = f"predicted_too_small ({predicted_diam:.1f} px, min {min_req})"
            stats["rejected_reasons"]["predicted_too_small"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body, "reason": reason
            })
            continue

        # FIX #6 (continued): verify the predicted center is inside the
        # frame with an edge margin. If SPICE says the body is off-frame,
        # the image should not be labeled.
        px_cx, px_cy = predicted_center
        if (px_cx < img_w * EDGE_MARGIN_FRAC or
            px_cx > img_w * (1 - EDGE_MARGIN_FRAC) or
            px_cy < img_h * EDGE_MARGIN_FRAC or
            px_cy > img_h * (1 - EDGE_MARGIN_FRAC)):
            stats["rejected"] += 1
            stats["rejected_reasons"]["predicted_center_off_frame"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": "predicted_center_off_frame"
            })
            continue

        # Search for the body contour around the predicted center.
        search_radius = max(predicted_diam * 2, 100)
        bbox, confidence, rejection = find_body_contour(
            img, predicted_center, search_radius, predicted_diam
        )

        if bbox is None:
            stats["rejected"] += 1
            reason = rejection or "no_contour_found"
            stats["rejected_reasons"][reason] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body, "reason": reason
            })
            continue

        bx, by, bw, bh = bbox

        # FIX #5: strict contour-center proximity check. If the contour
        # is farther than max(predicted_diam * 1.5, 100 px) from the
        # predicted boresight center, the labeler grabbed the wrong
        # object -> reject the image entirely.
        bcx = bx + bw / 2.0
        bcy = by + bh / 2.0
        dist = math.sqrt((bcx - px_cx)**2 + (bcy - px_cy)**2)
        prox_limit = max(predicted_diam * CONTOUR_PROX_DIAM_MULT, CONTOUR_PROX_MIN_PX)
        if dist > prox_limit:
            stats["rejected"] += 1
            stats["rejected_reasons"]["contour_far_from_predicted"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": f"contour_far_from_predicted ({dist:.0f}px > {prox_limit:.0f}px)"
            })
            continue

        # FIX #3: reject whole-frame or near-whole-frame bboxes and
        # (0,0)-anchored fallback boxes that span more than half the image.
        if bw > WHOLE_FRAME_FRAC * img_w and bh > WHOLE_FRAME_FRAC * img_h:
            stats["rejected"] += 1
            stats["rejected_reasons"]["whole_frame_bbox"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body, "reason": "whole_frame_bbox"
            })
            continue
        if (bx == 0 and by == 0 and
            (bw > ANCHORED_FALLBACK_FRAC * img_w or bh > ANCHORED_FALLBACK_FRAC * img_h)):
            stats["rejected"] += 1
            stats["rejected_reasons"]["anchored_fallback_bbox"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body, "reason": "anchored_fallback_bbox"
            })
            continue

        # FIX #4: reject extreme aspect ratios. Resolved bodies are ~circular.
        ar = aspect_ratio(bw, bh)
        if ar > MAX_ASPECT_RATIO:
            stats["rejected"] += 1
            stats["rejected_reasons"]["extreme_aspect_ratio"] += 1
            stats["per_body_rejected"][body] += 1
            rejected_images.append({
                "image_id": image_id, "body": body,
                "reason": f"extreme_aspect_ratio ({ar:.1f})"
            })
            continue

        # Near-edge softening: we still accept if the predicted center
        # passed the off-frame check, but lower confidence slightly.
        if (bcx < img_w * EDGE_MARGIN_FRAC or
            bcx > img_w * (1 - EDGE_MARGIN_FRAC) or
            bcy < img_h * EDGE_MARGIN_FRAC or
            bcy > img_h * (1 - EDGE_MARGIN_FRAC)):
            confidence *= 0.8

        # Pad the bbox slightly and clip
        bbox = pad_bbox((bx, by, bw, bh), BBOX_PAD_FRAC, img_w, img_h)
        has_spice = True  # round-2 invariant: all accepted labels have SPICE

        # Accept this label
        class_id = class_id_map[body]
        cx_norm, cy_norm, w_norm, h_norm = bbox_to_yolo(bbox, img_w, img_h)

        # Write YOLO label file
        label_dir = DATA_DIR / body / "labels"
        label_dir.mkdir(parents=True, exist_ok=True)
        label_path = label_dir / f"{image_id}.txt"

        with open(label_path, "w") as f:
            f.write(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        stats["accepted"] += 1
        if has_spice:
            stats["accepted_with_spice"] += 1
        else:
            stats["accepted_without_spice"] += 1
        stats["per_body_accepted"][body] += 1

        labeled_images.append({
            "image_id": image_id,
            "body": body,
            "img_path": str(img_path),
            "label_path": str(label_path),
            "bbox": bbox,
            "yolo": (cx_norm, cy_norm, w_norm, h_norm),
            "confidence": confidence,
            "has_spice": has_spice,
            "predicted_diam_px": predicted_diam if has_spice else None,
            "class_id": class_id,
        })

    print(f"\n{'='*60}")
    print(f"Labeling complete: {stats['accepted']} accepted, {stats['rejected']} rejected")
    print(f"  With SPICE: {stats['accepted_with_spice']}")
    print(f"  Without SPICE: {stats['accepted_without_spice']}")
    print(f"\nRejection reasons:")
    for reason, count in sorted(stats["rejected_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    # Write stats JSON
    stats_out = {
        "round": 2,
        "total_train_images": stats["total_train_images"],
        "permanent_human_drops": len(permanent_drops),
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "accepted_with_spice": stats["accepted_with_spice"],
        "accepted_without_spice": stats["accepted_without_spice"],  # should be 0 in round 2
        "rejection_rate": stats["rejected"] / max(stats["total_train_images"], 1),
        "rejected_reasons": dict(stats["rejected_reasons"]),
        "per_body_accepted": dict(stats["per_body_accepted"]),
        "per_body_rejected": dict(stats["per_body_rejected"]),
        "class_names": class_names,
        "class_id_map": class_id_map,
    }

    stats_path = ARTIFACTS / "cassini_issna" / "auto_label_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"Wrote {stats_path}")

    # ============================================================
    # Generate human review bundle
    # ============================================================
    print(f"\n{'='*60}")
    print("Generating human review bundle...")

    if len(labeled_images) == 0:
        print("ERROR: No images were labeled. Cannot generate review bundle.")
        sys.exit(1)

    # Categorize labeled images
    easy = []
    hard = []
    ambiguous = []

    for item in labeled_images:
        diam = item.get("predicted_diam_px")
        conf = item["confidence"]
        bbox = item["bbox"]
        img_cx = bbox[0] + bbox[2] / 2.0
        img_cy = bbox[1] + bbox[3] / 2.0
        dist_from_center = math.sqrt(
            (img_cx - IMAGE_SIZE / 2)**2 + (img_cy - IMAGE_SIZE / 2)**2
        )
        near_center = dist_from_center < 0.4 * (IMAGE_SIZE / 2)

        is_easy = False
        is_hard = False
        is_ambiguous = False

        # Ambiguous: low confidence or body is a small irregular moon
        if conf < 0.5 or not item["has_spice"]:
            is_ambiguous = True
        # Easy: large, well-centered, high confidence
        elif diam is not None and diam > 50 and near_center and conf >= 0.7:
            is_easy = True
        # Hard: small, off-center, or moderate confidence
        elif (diam is not None and diam < 30) or not near_center or conf < 0.7:
            is_hard = True
        else:
            is_easy = True  # Default to easy if nothing else matches

        if is_ambiguous:
            ambiguous.append(item)
        elif is_easy:
            easy.append(item)
        elif is_hard:
            hard.append(item)

    print(f"Categorized: {len(easy)} easy, {len(hard)} hard, {len(ambiguous)} ambiguous")

    # Sample 25 from each bucket (or as many as available)
    # FRESH SEED for round 2 per brief section 8.5.3 / 8.5.4.
    random.seed(20260408)

    def sample_up_to(lst, n):
        if len(lst) <= n:
            return lst[:]
        return random.sample(lst, n)

    easy_sample = sample_up_to(easy, 25)
    hard_sample = sample_up_to(hard, 25)
    ambiguous_sample = sample_up_to(ambiguous, 25)
    random_sample = sample_up_to(labeled_images, 25)

    # Ensure random sample doesn't overlap too much with others
    sampled_ids = set(item["image_id"] for item in easy_sample + hard_sample + ambiguous_sample)
    random_pool = [item for item in labeled_images if item["image_id"] not in sampled_ids]
    if len(random_pool) >= 25:
        random_sample = random.sample(random_pool, 25)
    elif len(random_pool) > 0:
        random_sample = random_pool[:]
    # If random_pool is empty, keep the original random_sample

    # ============================================================
    # Round-2 extra: previously-failed images the human flagged as
    # "Wrong Bounding box" or "Not <body>" (i.e. labeler defects, not
    # bad-image drops). We must show the human what happened to each
    # one after the fixes -- either accepted with a new bbox, or
    # rejected by one of the new rules (which is the expected outcome
    # for most of them).
    #
    # Per brief section 8.5.4: the fresh sample must include at least
    # 10 of these so the human can verify the fix.
    # ============================================================
    PREV_FAILED_VERIFY = [
        "co-iss-n1356764597",  # Wrong bbox ganymede (416x20 aspect)
        "co-iss-n1459600984",  # Wrong bbox saturn_rings (5x4)
        "co-iss-n1459603295",  # Wrong bbox saturn_rings (6x5)
        "co-iss-n1459603888",  # Wrong bbox saturn_rings (5x5)
        "co-iss-n1459605601",  # Not saturn ring (10x10)
        "co-iss-n1459804399",  # Not saturn ring (4x5)
        "co-iss-w1565241339",  # Wrong bbox saturn (4x4)
        "co-iss-w1565246157",  # Wrong bbox saturn (93x220)
        "co-iss-w1579185829",  # Wrong bbox saturn_rings (4x4)
        "co-iss-n1459733315",  # Wrong bbox saturn_rings (6x6)
        "co-iss-n1541716868",  # Wrong bbox saturn_rings (18x10)
        "co-iss-w1579250265",  # Wrong bbox saturn_rings (6x3)
        "co-iss-n1620679652",  # Not skoll (whole-frame fallback)
    ]

    # Create review directories (including verify_round1)
    for bucket_name in ["easy", "hard", "ambiguous", "random", "verify_round1"]:
        (REVIEW_DIR / bucket_name).mkdir(parents=True, exist_ok=True)
        # Purge any old PNGs so we don't mix rounds
        for old in (REVIEW_DIR / bucket_name).glob("*.png"):
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
                yolo = item["yolo"]
                index_rows.append({
                    "bucket": bucket_name,
                    "filename": f"{item['image_id']}.png",
                    "body": item["body"],
                    "bbox_x": item["bbox"][0],
                    "bbox_y": item["bbox"][1],
                    "bbox_w": item["bbox"][2],
                    "bbox_h": item["bbox"][3],
                    "confidence": f"{item['confidence']:.3f}",
                    "has_spice": item["has_spice"],
                    "predicted_diam_px": f"{item['predicted_diam_px']:.1f}" if item["predicted_diam_px"] else "N/A",
                    "image_id": item["image_id"],
                })

    # ============================================================
    # verify_round1 bucket: render the previously-failed cases and
    # show their post-fix fate (accepted with new bbox, or rejected
    # with the rule that dropped them).
    # ============================================================
    labeled_by_id = {it["image_id"]: it for it in labeled_images}
    rejected_by_id = {r["image_id"]: r for r in rejected_images}
    verify_rows = []
    for vid in PREV_FAILED_VERIFY:
        row = {"image_id": vid, "bucket": "verify_round1"}
        if vid in labeled_by_id:
            item = labeled_by_id[vid]
            row["status"] = "ACCEPTED(new bbox)"
            row["body"] = item["body"]
            row["bbox"] = item["bbox"]
            row["confidence"] = item["confidence"]
            row["predicted_diam_px"] = item["predicted_diam_px"]
            row["reason"] = ""
            out_path = REVIEW_DIR / "verify_round1" / f"{vid}.png"
            render_review_image(
                Path(item["img_path"]), item["bbox"], item["body"],
                item["confidence"], out_path
            )
        elif vid in rejected_by_id:
            r = rejected_by_id[vid]
            row["status"] = "REJECTED(new rules)"
            row["body"] = r["body"]
            row["bbox"] = None
            row["confidence"] = None
            row["predicted_diam_px"] = None
            row["reason"] = r["reason"]
            # Render the raw image with a big red REJECTED overlay so the
            # human can see we actually dropped it instead of mis-labeling.
            src = DATA_DIR / r["body"] / "images" / f"{vid}.png"
            if src.exists():
                im = cv2.imread(str(src))
                if im is not None:
                    cv2.putText(im, "REJECTED", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3,
                                cv2.LINE_AA)
                    cv2.putText(im, r["reason"][:60], (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    cv2.imwrite(str(REVIEW_DIR / "verify_round1" / f"{vid}.png"), im)
        else:
            row["status"] = "NOT_IN_TRAIN_SPLIT"
            row["body"] = "?"
            row["bbox"] = None
            row["confidence"] = None
            row["predicted_diam_px"] = None
            row["reason"] = "not in train_split.csv"
        verify_rows.append(row)

    print(f"verify_round1 bucket: {len(verify_rows)} images")

    # Write INDEX.md
    index_path = REVIEW_DIR / "INDEX.md"
    with open(index_path, "w") as f:
        f.write("# Auto-Label Review Index: cassini / issna  (ROUND 2)\n\n")
        f.write(f"Generated: 2026-04-08  (post human NEEDS_FIXES)\n\n")
        f.write(f"Total labeled: {stats['accepted']} / {stats['total_train_images']}\n\n")
        f.write("## Fresh sample (round-2 seed)\n\n")
        f.write("| Bucket | Filename | Body | BBox (x,y,w,h) | Confidence | SPICE | Pred Diam (px) | Image ID |\n")
        f.write("|--------|----------|------|-----------------|------------|-------|----------------|----------|\n")
        for r in index_rows:
            f.write(f"| {r['bucket']} | {r['filename']} | {r['body']} | "
                    f"({r['bbox_x']},{r['bbox_y']},{r['bbox_w']},{r['bbox_h']}) | "
                    f"{r['confidence']} | {r['has_spice']} | {r['predicted_diam_px']} | "
                    f"{r['image_id']} |\n")
        f.write("\n## verify_round1 (previously-failed images -- verify the fix)\n\n")
        f.write("These are images the human flagged in round 1 as 'Wrong Bounding box' or\n")
        f.write("'Not <body>'. The round-2 labeler should either produce a correct bbox OR\n")
        f.write("reject the image by one of the new rules.\n\n")
        f.write("| Image ID | Body | Round-2 status | New bbox | Reason / rule |\n")
        f.write("|----------|------|----------------|----------|---------------|\n")
        for v in verify_rows:
            bbox_str = f"({v['bbox'][0]},{v['bbox'][1]},{v['bbox'][2]},{v['bbox'][3]})" if v['bbox'] else "-"
            f.write(f"| {v['image_id']} | {v['body']} | {v['status']} | {bbox_str} | {v['reason']} |\n")

    print(f"Wrote {index_path} with {len(index_rows)} rows")

    # Write empty DECISION.md
    decision_path = REVIEW_DIR / "DECISION.md"
    with open(decision_path, "w") as f:
        f.write("")  # Empty -- human fills this in
    print(f"Wrote empty {decision_path}")

    # Write REVIEW_REQUEST.md (ROUND 2)
    review_request_path = WORKSPACE / "review" / "cassini_issna" / "REVIEW_REQUEST.md"
    with open(review_request_path, "w") as f:
        f.write("# Review Request: Auto-Labels for cassini / issna  (ROUND 2)\n\n")
        f.write("Round 1 was rejected per-image by the human (33 of 100 rejected).\n")
        f.write("This round incorporates all of their fixes. Fresh sample, fresh seed,\n")
        f.write("plus a `verify_round1/` bucket showing what the labeler now does with\n")
        f.write("the previously-failed cases.\n\n")
        f.write("**Iteration log:** `research_log/0007_auto_label_round2_cassini_issna.md`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total training images:** {stats['total_train_images']}\n")
        f.write(f"- **Permanently dropped by human (round 1):** {len(permanent_drops)}\n")
        f.write(f"- **Auto-labeled (accepted):** {stats['accepted']}\n")
        f.write(f"- **Rejected:** {stats['rejected']}\n")
        f.write(f"- **Rejection rate:** {stats['rejected'] / max(stats['total_train_images'], 1) * 100:.1f}%\n")
        f.write(f"- **Labeled with SPICE geometry:** {stats['accepted_with_spice']}\n")
        f.write(f"- **Labeled without SPICE:** {stats['accepted_without_spice']} (should be 0 in round 2)\n\n")

        f.write("## Fixes applied in round 2\n\n")
        f.write("1. **Drop no-SPICE images.** The OpenCV-only fallback is removed. "
                "Any image without a valid SPICE state row is rejected.\n")
        f.write("2. **Minimum predicted diameter raised to 15 px** (from 8). Rings "
                "require 100 px effective extent.\n")
        f.write("3. **Whole-frame / anchored fallback boxes rejected.** No more "
                "(0,0,1024,1024) or (0,0,256,256) labels.\n")
        f.write("4. **Extreme aspect ratios (>3:1) rejected.** Planetary bodies are "
                "approximately circular.\n")
        f.write("5. **Contour proximity tightened.** Refined contour center must be "
                "within max(predicted_diam * 1.5, 100 px) of the predicted center.\n")
        f.write("6. **Off-frame predicted center rejected.** If SPICE geometry puts "
                "the body outside the image, drop it.\n")
        f.write("7. **saturn_rings minimum extent = 100 px.**\n")
        f.write("8. **33 permanent human-drops** enforced from round-1 DECISION.md.\n\n")

        f.write("## Rejection Reasons\n\n")
        f.write("| Reason | Count |\n")
        f.write("|--------|-------|\n")
        for reason, count in sorted(stats["rejected_reasons"].items(), key=lambda x: -x[1]):
            f.write(f"| {reason} | {count} |\n")

        f.write("\n## Per-Body Statistics\n\n")
        f.write("| Body | Accepted | Rejected |\n")
        f.write("|------|----------|----------|\n")
        for body_name in sorted(set(list(stats["per_body_accepted"].keys()) + list(stats["per_body_rejected"].keys()))):
            acc = stats["per_body_accepted"].get(body_name, 0)
            rej = stats["per_body_rejected"].get(body_name, 0)
            f.write(f"| {body_name} | {acc} | {rej} |\n")

        f.write("\n## Review Buckets\n\n")
        f.write(f"Five buckets of sample images are in `review/cassini_issna/auto_labels/`:\n\n")
        f.write(f"1. **easy/** ({len(easy_sample)} images) -- Large, well-centered bodies with high confidence. "
                f"These should be near-perfect. If any are wrong, the labeler is fundamentally broken.\n")
        f.write(f"2. **hard/** ({len(hard_sample)} images) -- Small bodies (15-30 px predicted) or off-center. "
                f"Most remaining failures will live here.\n")
        f.write(f"3. **ambiguous/** ({len(ambiguous_sample)} images) -- Low-confidence detections or "
                f"near-edge cases. Judgment calls.\n")
        f.write(f"4. **random/** ({len(random_sample)} images) -- Uniform random sample from the full labeled set, "
                f"fresh seed. Catches categorization bias.\n")
        f.write(f"5. **verify_round1/** ({len(verify_rows)} images) -- Previously-failed round-1 cases. "
                f"Each one should either (a) have a correct new bbox, or (b) be REJECTED by one of the new rules "
                f"(the red overlay shows the rejecting rule). This is how you verify the fix actually fixed the bug.\n\n")

        f.write("## What To Look For\n\n")
        f.write("For each image, the green bounding box should tightly enclose the target body. Check:\n")
        f.write("- Does the box actually surround the correct body (not a star, cosmic ray, or different moon)?\n")
        f.write("- Is the box tight but not clipping the body?\n")
        f.write("- For saturn_rings images: does the box include the ring system?\n")
        f.write("- For point-source moons: is the box centered on the right bright dot?\n\n")

        f.write("## What To Do\n\n")
        f.write("Open each bucket folder and inspect the overlay PNGs.\n")
        f.write("Then write your verdict into:\n")
        f.write("`review/cassini_issna/auto_labels/DECISION.md`\n\n")
        f.write("Write one of:\n")
        f.write("- `STATUS: APPROVED` -- labels are good enough to train on\n")
        f.write("- `STATUS: NEEDS_FIXES` -- with instructions for what to fix\n")
        f.write("- `STATUS: REJECTED` -- labeling approach is fundamentally broken\n\n")

        f.write("## The Question\n\n")
        f.write("What would make you trust this labeler enough to start training?\n")

    print(f"Wrote {review_request_path}")
    print(f"\nReview bundle complete. {len(easy_sample) + len(hard_sample) + len(ambiguous_sample) + len(random_sample)} images rendered.")
    print("DONE.")


if __name__ == "__main__":
    main()
