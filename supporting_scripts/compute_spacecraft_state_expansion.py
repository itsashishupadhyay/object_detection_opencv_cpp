#!/usr/bin/env python3
"""
EXPANSION: Compute spacecraft state for new Cassini ISS-NA images.

Reads the existing spacecraft_state.csv, identifies manifest rows that
have no state row yet, computes state for those, and APPENDS to the CSV.
Updates image_manifest.csv state_verified for newly computed rows only.

Author: Verification sub-agent (expansion iteration)
Date: 2026-04-10
"""

import csv
import hashlib
import json
import os
import sys
import glob
import traceback
from datetime import datetime, timezone

WORKSPACE = "/Users/upadhyay/dev/ICES/object_detection_opencv_cpp"
VENV_SITE = os.path.join(WORKSPACE, ".yolo_venv/lib/python3.12/site-packages")
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import spiceypy as spice

SPICE_CACHE = os.path.join(WORKSPACE, "spice_cache")
MANIFEST_PATH = os.path.join(WORKSPACE, "artifacts/cassini_issna/image_manifest.csv")
OUTPUT_PATH = os.path.join(WORKSPACE, "artifacts/spacecraft_state.csv")
CASSINI_NAIF_ID = -82
CASSINI_ISS_NAC_ID = -82360

BODY_NAME_MAP = {
    "saturn": "SATURN",
    "saturn_rings": "SATURN",
    "titan": "TITAN",
    "enceladus": "ENCELADUS",
    "mimas": "MIMAS",
    "tethys": "TETHYS",
    "dione": "DIONE",
    "rhea": "RHEA",
    "hyperion": "HYPERION",
    "iapetus": "IAPETUS",
    "phoebe": "PHOEBE",
    "janus": "JANUS",
    "epimetheus": "EPIMETHEUS",
    "prometheus": "PROMETHEUS",
    "pandora": "PANDORA",
    "atlas": "ATLAS",
    "pan": "PAN",
    "helene": "HELENE",
    "telesto": "TELESTO",
    "calypso": "CALYPSO",
    "polydeuces": "POLYDEUCES",
    "methone": "METHONE",
    "pallene": "PALLENE",
    "anthe": "ANTHE",
    "aegaeon": "AEGAEON",
    "daphnis": "DAPHNIS",
    "jupiter": "JUPITER",
    "io": "IO",
    "europa": "EUROPA",
    "ganymede": "GANYMEDE",
    "callisto": "CALLISTO",
    "himalia": "HIMALIA",
    "earth": "EARTH",
    "moon": "MOON",
    "venus": "VENUS",
    "pluto": "PLUTO",
    "albiorix": "ALBIORIX",
    "bebhionn": "BEBHIONN",
    "bergelmir": "BERGELMIR",
    "bestla": "BESTLA",
    "erriapus": "ERRIAPUS",
    "fornjot": "FORNJOT",
    "greip": "GREIP",
    "hati": "HATI",
    "hyrrokkin": "HYRROKKIN",
    "ijiraq": "IJIRAQ",
    "jarnsaxa": "JARNSAXA",
    "kari": "KARI",
    "kiviuq": "KIVIUQ",
    "loge": "LOGE",
    "mundilfari": "MUNDILFARI",
    "narvi": "NARVI",
    "paaliaq": "PAALIAQ",
    "skathi": "SKATHI",
    "skoll": "SKOLL",
    "siarnaq": "SIARNAQ",
    "surtur": "SURTUR",
    "suttungr": "SUTTUNGR",
    "tarqeq": "TARQEQ",
    "tarvos": "TARVOS",
    "thrymr": "THRYMR",
    "ymir": "YMIR",
    "s_2004_s_12": "S/2004 S 12",
    "s_2004_s_13": "S/2004 S 13",
    "unknown": None,
}

BARYCENTER_FALLBACK = {
    "SATURN": "SATURN BARYCENTER",
    "JUPITER": "JUPITER BARYCENTER",
    "EARTH": "EARTH BARYCENTER",
    "VENUS": "VENUS BARYCENTER",
    "MARS": "MARS BARYCENTER",
    "PLUTO": "PLUTO BARYCENTER",
    "NEPTUNE": "NEPTUNE BARYCENTER",
    "URANUS": "URANUS BARYCENTER",
    "MERCURY": "MERCURY BARYCENTER",
}


def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_kernels():
    kernel_files = []
    text_kernels = [
        "naif0012.tls",
        "cas00172.tsc",
        "cas_v43.tf",
        "cas_iss_v10.ti",
        "pck00011.tpc",
    ]
    for k in text_kernels:
        path = os.path.join(SPICE_CACHE, k)
        if os.path.exists(path):
            spice.furnsh(path)
            kernel_files.append(k)
            print(f"  Loaded: {k}")
        else:
            print(f"  WARNING: Missing kernel {k}")

    spk_files = sorted(glob.glob(os.path.join(SPICE_CACHE, "*.bsp")))
    for path in spk_files:
        name = os.path.basename(path)
        spice.furnsh(path)
        kernel_files.append(name)
        print(f"  Loaded SPK: {name}")

    ck_files = sorted(glob.glob(os.path.join(SPICE_CACHE, "*rc.bc")))
    for path in ck_files:
        name = os.path.basename(path)
        spice.furnsh(path)
        kernel_files.append(name)
        print(f"  Loaded CK: {name}")

    return kernel_files


def compute_state_for_image(timestamp_utc, target_body_spice):
    result = {}
    try:
        et = spice.utc2et(timestamp_utc)
    except Exception as e:
        return None, f"utc2et failed: {e}"

    targets_to_try = [target_body_spice]
    if target_body_spice in BARYCENTER_FALLBACK:
        targets_to_try.append(BARYCENTER_FALLBACK[target_body_spice])

    spk_success = False
    last_error = None
    for target in targets_to_try:
        try:
            state, lt = spice.spkezr("CASSINI", et, "J2000", "LT+S", target)
            result["position_km_x"] = state[0]
            result["position_km_y"] = state[1]
            result["position_km_z"] = state[2]
            result["velocity_kms_x"] = state[3]
            result["velocity_kms_y"] = state[4]
            result["velocity_kms_z"] = state[5]
            result["position_frame"] = "J2000"
            spk_success = True
            break
        except Exception as e:
            last_error = e

    if not spk_success:
        return None, f"spkezr failed for {target_body_spice}: {last_error}"

    try:
        rot_matrix = spice.pxform("J2000", "CASSINI_ISS_NAC", et)
        quat = spice.m2q(rot_matrix)
        result["attitude_q_w"] = quat[0]
        result["attitude_q_x"] = quat[1]
        result["attitude_q_y"] = quat[2]
        result["attitude_q_z"] = quat[3]
        result["attitude_frame"] = "J2000_to_CASSINI_ISS_NAC"
    except Exception:
        result["attitude_q_w"] = ""
        result["attitude_q_x"] = ""
        result["attitude_q_y"] = ""
        result["attitude_q_z"] = ""
        result["attitude_frame"] = "NO_CK_COVERAGE"

    return result, None


def main():
    print("=" * 70)
    print("EXPANSION: Cassini ISS-NA Spacecraft State Computation")
    print("=" * 70)

    # Step 1: Read existing state rows to know what's already computed
    existing_ids = set()
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["image_id"])
    print(f"\nExisting spacecraft_state rows: {len(existing_ids)}")

    # Step 2: Read manifest and identify rows needing state computation
    all_manifest_rows = []
    new_images = []
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        manifest_fieldnames = reader.fieldnames
        for row in reader:
            all_manifest_rows.append(row)
            if row["image_id"] not in existing_ids:
                new_images.append(row)
    print(f"Total manifest rows: {len(all_manifest_rows)}")
    print(f"New images needing state: {len(new_images)}")

    # Step 3: Load SPICE kernels
    print("\nLoading SPICE kernels...")
    kernel_files = load_kernels()
    print(f"Loaded {len(kernel_files)} kernels total.\n")

    # Step 4: Compute kernel hashes
    print("Computing kernel SHA-256 hashes...")
    kernel_hashes = {}
    for k in kernel_files:
        path = os.path.join(SPICE_CACHE, k)
        if os.path.exists(path):
            kernel_hashes[k] = sha256_file(path)
    kernel_hash_str = "; ".join(f"{k}:{v[:16]}" for k, v in sorted(kernel_hashes.items()))
    kernel_list_str = "; ".join(sorted(kernel_files))
    print("Done.\n")

    # Step 5: Compute state for each new image
    new_state_rows = []
    full_state_count = 0
    position_only_count = 0
    fail_count = 0
    fail_reasons = {}
    fail_by_body = {}
    success_by_body = {}
    queried_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"Computing spacecraft states for {len(new_images)} new images...")
    for i, img in enumerate(new_images):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(new_images)}...")

        image_id = img["image_id"]
        timestamp_utc = img["timestamp_utc"]
        body = img["body"].strip().lower()
        target_meta = img.get("target_body_from_metadata", "")

        target_spice = BODY_NAME_MAP.get(body)
        if target_spice is None:
            fail_count += 1
            reason = f"unknown body: {body}"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            fail_by_body[body] = fail_by_body.get(body, 0) + 1
            continue

        result, error = compute_state_for_image(timestamp_utc, target_spice)
        if result is None:
            fail_count += 1
            err_key = error.split(":")[0] if error else "unknown"
            fail_reasons[err_key] = fail_reasons.get(err_key, 0) + 1
            fail_by_body[body] = fail_by_body.get(body, 0) + 1
            if fail_count <= 20:
                print(f"  FAIL [{image_id}]: {error}")
            continue

        if result.get("attitude_frame") == "NO_CK_COVERAGE":
            position_only_count += 1
        else:
            full_state_count += 1

        success_by_body[body] = success_by_body.get(body, 0) + 1

        row = {
            "mission": "cassini",
            "image_id": image_id,
            "timestamp_utc": timestamp_utc,
            "position_km_x": f"{result['position_km_x']:.6f}",
            "position_km_y": f"{result['position_km_y']:.6f}",
            "position_km_z": f"{result['position_km_z']:.6f}",
            "position_frame": result["position_frame"],
            "velocity_kms_x": f"{result['velocity_kms_x']:.6f}",
            "velocity_kms_y": f"{result['velocity_kms_y']:.6f}",
            "velocity_kms_z": f"{result['velocity_kms_z']:.6f}",
            "attitude_q_w": f"{result['attitude_q_w']:.10f}" if result["attitude_q_w"] != "" else "",
            "attitude_q_x": f"{result['attitude_q_x']:.10f}" if result["attitude_q_x"] != "" else "",
            "attitude_q_y": f"{result['attitude_q_y']:.10f}" if result["attitude_q_y"] != "" else "",
            "attitude_q_z": f"{result['attitude_q_z']:.10f}" if result["attitude_q_z"] != "" else "",
            "attitude_frame": result["attitude_frame"],
            "target_body_from_metadata": target_meta,
            "spice_kernels_used": kernel_list_str,
            "kernel_sha256": kernel_hash_str,
            "queried_at": queried_at,
        }
        new_state_rows.append(row)

    print(f"\n{'=' * 70}")
    print(f"EXPANSION RESULTS:")
    print(f"  New images processed: {len(new_images)}")
    print(f"  Full state (pos + att): {full_state_count}")
    print(f"  Position only (no CK): {position_only_count}")
    print(f"  Failed: {fail_count}")
    if fail_reasons:
        print(f"\nFailure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"\nSuccess by body:")
    for body, count in sorted(success_by_body.items(), key=lambda x: -x[1]):
        print(f"  {body}: {count}")
    if fail_by_body:
        print(f"\nFailures by body:")
        for body, count in sorted(fail_by_body.items(), key=lambda x: -x[1]):
            print(f"  {body}: {count}")
    print(f"{'=' * 70}\n")

    # Step 6: APPEND new rows to spacecraft_state.csv
    fieldnames = [
        "mission", "image_id", "timestamp_utc",
        "position_km_x", "position_km_y", "position_km_z", "position_frame",
        "velocity_kms_x", "velocity_kms_y", "velocity_kms_z",
        "attitude_q_w", "attitude_q_x", "attitude_q_y", "attitude_q_z",
        "attitude_frame",
        "target_body_from_metadata",
        "spice_kernels_used", "kernel_sha256", "queried_at",
    ]

    with open(OUTPUT_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        # Don't write header - file already has one
        for row in new_state_rows:
            writer.writerow(row)
    print(f"Appended {len(new_state_rows)} rows to {OUTPUT_PATH}")

    # Step 7: Update manifest - only change rows that were newly computed
    new_full_ids = set(r["image_id"] for r in new_state_rows
                       if r["attitude_frame"] != "NO_CK_COVERAGE")
    new_pos_ids = set(r["image_id"] for r in new_state_rows
                      if r["attitude_frame"] == "NO_CK_COVERAGE")

    updated_count = 0
    for row in all_manifest_rows:
        if row["image_id"] in new_full_ids:
            row["state_verified"] = "true"
            updated_count += 1
        elif row["image_id"] in new_pos_ids:
            row["state_verified"] = "position_only"
            updated_count += 1
        # else: leave existing value unchanged

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fieldnames)
        writer.writeheader()
        for row in all_manifest_rows:
            writer.writerow(row)
    print(f"Updated manifest: {updated_count} rows changed")

    spice.kclear()

    # Output JSON summary for the iteration log
    summary = {
        "new_images_total": len(new_images),
        "full_state": full_state_count,
        "position_only": position_only_count,
        "failed": fail_count,
        "fail_reasons": fail_reasons,
        "fail_by_body": fail_by_body,
        "success_by_body": success_by_body,
        "new_rows_appended": len(new_state_rows),
        "total_state_rows_after": len(existing_ids) + len(new_state_rows),
        "kernels_loaded": len(kernel_files),
    }
    print("\nJSON_SUMMARY_START")
    print(json.dumps(summary, indent=2))
    print("JSON_SUMMARY_END")

    return summary


if __name__ == "__main__":
    main()
