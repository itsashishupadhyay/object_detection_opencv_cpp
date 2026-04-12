#!/usr/bin/env python3
"""
Compute verified spacecraft state records for Cassini ISS-NA images.

For each image in the manifest, queries SPICE kernels for:
- Spacecraft position and velocity relative to the target body (J2000 frame)
- Spacecraft attitude quaternion from CK kernels
- Target body position from the spacecraft

Outputs artifacts/spacecraft_state.csv with one row per successfully resolved image.
Updates image_manifest.csv to set state_verified = true for successful lookups.

Author: Verification sub-agent
Date: 2026-04-08
"""

import csv
import hashlib
import os
import sys
import glob
import traceback
from datetime import datetime, timezone

# Add venv to path
WORKSPACE = "/Users/upadhyay/dev/ICES/object_detection_opencv_cpp"
VENV_SITE = os.path.join(WORKSPACE, ".yolo_venv/lib/python3.12/site-packages")
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import spiceypy as spice

# ----- Configuration -----
SPICE_CACHE = os.path.join(WORKSPACE, "spice_cache")
MANIFEST_PATH = os.path.join(WORKSPACE, "artifacts/cassini_issna/image_manifest.csv")
OUTPUT_PATH = os.path.join(WORKSPACE, "artifacts/spacecraft_state.csv")
CASSINI_NAIF_ID = -82
CASSINI_ISS_NAC_ID = -82360

# Map target body names to NAIF IDs / SPICE names
# We'll use spice.bodn2c() for the lookup, but some names in the manifest
# need mapping to SPICE-recognized names.
BODY_NAME_MAP = {
    "saturn": "SATURN",
    "saturn_rings": "SATURN",  # Use Saturn barycenter for rings
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
    # Irregular satellites - may not be in standard ephemeris
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
    "unknown": None,  # Can't compute geometry for unknown targets
}


def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_kernels():
    """Load all SPICE kernels from the cache directory."""
    # Order matters: load text kernels first, then binary
    kernel_files = []

    # Text kernels (order: LSK, SCLK, FK, IK, PCK)
    text_kernels = [
        "naif0012.tls",      # LSK
        "cas00172.tsc",      # SCLK
        "cas_v43.tf",        # FK
        "cas_iss_v10.ti",    # IK
        "pck00011.tpc",      # PCK
    ]

    for k in text_kernels:
        path = os.path.join(SPICE_CACHE, k)
        if os.path.exists(path):
            spice.furnsh(path)
            kernel_files.append(k)
            print(f"  Loaded: {k}")
        else:
            print(f"  WARNING: Missing kernel {k}")

    # SPK kernels (binary)
    spk_files = sorted(glob.glob(os.path.join(SPICE_CACHE, "*.bsp")))
    for path in spk_files:
        name = os.path.basename(path)
        spice.furnsh(path)
        kernel_files.append(name)
        print(f"  Loaded SPK: {name}")

    # CK kernels (binary) - load all reconstructed CKs
    ck_files = sorted(glob.glob(os.path.join(SPICE_CACHE, "*rc.bc")))
    for path in ck_files:
        name = os.path.basename(path)
        spice.furnsh(path)
        kernel_files.append(name)
        print(f"  Loaded CK: {name}")

    return kernel_files


def compute_kernel_hashes(kernel_files):
    """Compute SHA-256 hashes for all loaded kernels."""
    hashes = {}
    for k in kernel_files:
        path = os.path.join(SPICE_CACHE, k)
        if os.path.exists(path):
            hashes[k] = sha256_file(path)
    return hashes


def resolve_target_body(body_name_lower):
    """Resolve a body name from the manifest to a SPICE-compatible name."""
    if body_name_lower not in BODY_NAME_MAP:
        return None
    return BODY_NAME_MAP[body_name_lower]


    # Fallback mapping: body name -> barycenter name for when direct lookup fails
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


def compute_state_for_image(timestamp_utc, target_body_spice):
    """
    Compute spacecraft state for a single image.

    Returns dict with position, velocity, attitude, or None on failure.
    If the primary target body lookup fails, tries the barycenter as fallback
    (appropriate for distant observations where body center ~ barycenter).
    """
    result = {}

    # Convert UTC timestamp to ephemeris time (ET)
    try:
        et = spice.utc2et(timestamp_utc)
    except Exception as e:
        return None, f"utc2et failed: {e}"

    # --- Position and velocity ---
    # Query spacecraft position relative to target body in J2000
    # Try primary target first, then fall back to barycenter
    targets_to_try = [target_body_spice]
    if target_body_spice in BARYCENTER_FALLBACK:
        targets_to_try.append(BARYCENTER_FALLBACK[target_body_spice])

    spk_success = False
    used_target = None
    last_error = None
    for target in targets_to_try:
        try:
            state, lt = spice.spkezr(
                "CASSINI",
                et,
                "J2000",
                "LT+S",  # Light-time + stellar aberration correction
                target
            )
            result["position_km_x"] = state[0]
            result["position_km_y"] = state[1]
            result["position_km_z"] = state[2]
            result["velocity_kms_x"] = state[3]
            result["velocity_kms_y"] = state[4]
            result["velocity_kms_z"] = state[5]
            result["position_frame"] = "J2000"
            spk_success = True
            used_target = target
            break
        except Exception as e:
            last_error = e

    if not spk_success:
        return None, f"spkezr failed for {target_body_spice}: {last_error}"

    # --- Attitude quaternion ---
    # Get rotation matrix from J2000 to CASSINI_ISS_NAC frame
    try:
        rot_matrix = spice.pxform("J2000", "CASSINI_ISS_NAC", et)
        # Convert rotation matrix to quaternion (SPICE convention: w, x, y, z)
        quat = spice.m2q(rot_matrix)
        result["attitude_q_w"] = quat[0]
        result["attitude_q_x"] = quat[1]
        result["attitude_q_y"] = quat[2]
        result["attitude_q_z"] = quat[3]
        result["attitude_frame"] = "J2000_to_CASSINI_ISS_NAC"
    except Exception as e:
        # Attitude not available (CK gap) - still report position
        result["attitude_q_w"] = ""
        result["attitude_q_x"] = ""
        result["attitude_q_y"] = ""
        result["attitude_q_z"] = ""
        result["attitude_frame"] = "NO_CK_COVERAGE"

    return result, None


def main():
    print("=" * 70)
    print("Cassini ISS-NA Spacecraft State Computation")
    print("=" * 70)
    print()

    # Load kernels
    print("Loading SPICE kernels...")
    kernel_files = load_kernels()
    print(f"\nLoaded {len(kernel_files)} kernels total.\n")

    # Compute kernel hashes
    print("Computing kernel SHA-256 hashes...")
    kernel_hashes = compute_kernel_hashes(kernel_files)
    kernel_hash_str = "; ".join(f"{k}:{v[:16]}" for k, v in sorted(kernel_hashes.items()))
    kernel_list_str = "; ".join(sorted(kernel_files))
    print(f"Done.\n")

    # Read manifest
    print(f"Reading manifest from {MANIFEST_PATH}...")
    images = []
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            images.append(row)
    print(f"Found {len(images)} images.\n")

    # Process each image
    state_rows = []
    success_count = 0
    fail_count = 0
    attitude_fail_count = 0
    fail_reasons = {}
    queried_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    print("Computing spacecraft states...")
    for i, img in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(images)}...")

        image_id = img["image_id"]
        timestamp_utc = img["timestamp_utc"]
        body = img["body"].strip().lower()
        target_meta = img.get("target_body_from_metadata", "")

        # Resolve target body
        target_spice = resolve_target_body(body)
        if target_spice is None:
            fail_count += 1
            reason = f"unknown body: {body}"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            continue

        # Compute state
        result, error = compute_state_for_image(timestamp_utc, target_spice)
        if result is None:
            fail_count += 1
            # Simplify error for counting
            err_key = error.split(":")[0] if error else "unknown"
            fail_reasons[err_key] = fail_reasons.get(err_key, 0) + 1
            if fail_count <= 20:
                print(f"  FAIL [{image_id}]: {error}")
            continue

        # Track attitude failures separately
        if result.get("attitude_frame") == "NO_CK_COVERAGE":
            attitude_fail_count += 1

        success_count += 1

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
        state_rows.append(row)

    print(f"\n{'=' * 70}")
    print(f"Results: {success_count} succeeded, {fail_count} failed")
    print(f"  Of successes, {attitude_fail_count} lack attitude (CK gap)")
    print(f"  Fully resolved (pos + att): {success_count - attitude_fail_count}")
    if fail_reasons:
        print(f"\nFailure breakdown:")
        for reason, count in sorted(fail_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")
    print(f"{'=' * 70}\n")

    # Write spacecraft_state.csv
    fieldnames = [
        "mission", "image_id", "timestamp_utc",
        "position_km_x", "position_km_y", "position_km_z", "position_frame",
        "velocity_kms_x", "velocity_kms_y", "velocity_kms_z",
        "attitude_q_w", "attitude_q_x", "attitude_q_y", "attitude_q_z",
        "attitude_frame",
        "target_body_from_metadata",
        "spice_kernels_used", "kernel_sha256", "queried_at",
    ]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in state_rows:
            writer.writerow(row)

    print(f"Wrote {len(state_rows)} rows to {OUTPUT_PATH}")

    # Update manifest: set state_verified for successful lookups
    verified_ids = set(r["image_id"] for r in state_rows
                       if r["attitude_frame"] != "NO_CK_COVERAGE")
    position_only_ids = set(r["image_id"] for r in state_rows
                            if r["attitude_frame"] == "NO_CK_COVERAGE")

    updated_manifest = []
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        fieldnames_m = reader.fieldnames
        for row in reader:
            if row["image_id"] in verified_ids:
                row["state_verified"] = "true"
            elif row["image_id"] in position_only_ids:
                row["state_verified"] = "position_only"
            # else leave as false
            updated_manifest.append(row)

    with open(MANIFEST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_m)
        writer.writeheader()
        for row in updated_manifest:
            writer.writerow(row)

    print(f"Updated manifest: {len(verified_ids)} fully verified, "
          f"{len(position_only_ids)} position-only, "
          f"{len(images) - len(verified_ids) - len(position_only_ids)} unverified")

    # Cleanup SPICE
    spice.kclear()

    # Summary for iteration log
    print(f"\n{'=' * 70}")
    print("SUMMARY FOR ITERATION LOG:")
    print(f"  Total images: {len(images)}")
    print(f"  State computed (full): {success_count - attitude_fail_count}")
    print(f"  State computed (position only, no attitude): {attitude_fail_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Kernels loaded: {len(kernel_files)}")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"{'=' * 70}")

    return success_count, fail_count, attitude_fail_count, fail_reasons, kernel_hashes


if __name__ == "__main__":
    main()
