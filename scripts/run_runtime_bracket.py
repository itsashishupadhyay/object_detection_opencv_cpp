#!/usr/bin/env python3
"""Run the trained C++ binary over the cassini/issna test bracket.

- For each test_split image, invoke ./build/opencv_cpp_release --nav-decision
- Save the per-image JSON to artifacts/cassini_issna/decisions/<image_id>.json
- Resumable: skip if the JSON already exists.
- Parallel: up to 8 workers.
"""

import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path("/Users/upadhyay/dev/ICES/object_detection_opencv_cpp")
BINARY = ROOT / "build" / "opencv_cpp_release"
MODEL = ROOT / "weight" / "cassini_issna_planets.onnx"
LABELS = ROOT / "weight" / "cassini_issna_planets.names"
MANIFEST = ROOT / "artifacts" / "cassini_issna" / "image_manifest.csv"
TEST_SPLIT = ROOT / "artifacts" / "cassini_issna" / "test_split.csv"
OUTDIR = ROOT / "artifacts" / "cassini_issna" / "decisions"
ADJ_OUTDIR = ROOT / "artifacts" / "cassini_issna" / "decisions_adjacency_train"
ADJACENCY = ROOT / "artifacts" / "cassini_issna" / "adjacency_pairs.csv"


def load_manifest():
    with MANIFEST.open() as f:
        rdr = csv.DictReader(f)
        return {r["image_id"]: r for r in rdr}


def run_one(args):
    image_id, image_path, outdir = args
    out_json = Path(outdir) / f"{image_id}.json"
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text())
            return image_id, data.get("status", "UNKNOWN"), "cached"
        except Exception:
            pass  # fall through and re-run

    cmd = [
        str(BINARY),
        "--nav-decision",
        "--mission", "cassini",
        "--instrument", "issna",
        "-p", str(image_path),
        "-l", str(LABELS),
        "-m", str(MODEL),
    ]

    last_err = ""
    for attempt in range(3):
        try:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )
            if res.returncode != 0:
                last_err = f"rc={res.returncode}: {res.stderr.strip()[:500]}"
                continue
            # stdout is JSON object followed by summary line
            out = res.stdout.strip()
            # Find first '{' and matching '}' block
            try:
                # parse progressively
                end = out.rfind("}")
                if end < 0:
                    last_err = "no JSON braces"
                    continue
                data = json.loads(out[: end + 1])
            except json.JSONDecodeError as e:
                last_err = f"json_decode: {e}"
                continue
            out_json.write_text(json.dumps(data, indent=2))
            return image_id, data.get("status", "UNKNOWN"), "ok"
        except subprocess.TimeoutExpired:
            last_err = "timeout"
        except Exception as e:
            last_err = f"exc: {e}"

    # REFUSED placeholder
    refused = {
        "status": "REFUSED",
        "image_id": image_id,
        "mission": "cassini",
        "instrument": "issna",
        "refused_reason": f"binary_failure: {last_err}",
    }
    out_json.write_text(json.dumps(refused, indent=2))
    return image_id, "REFUSED", f"err:{last_err[:80]}"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    manifest = load_manifest()

    if mode == "test":
        outdir = OUTDIR
        outdir.mkdir(parents=True, exist_ok=True)
        with TEST_SPLIT.open() as f:
            rdr = csv.DictReader(f)
            ids = [r["image_id"] for r in rdr]
    elif mode == "adjacency_train":
        outdir = ADJ_OUTDIR
        outdir.mkdir(parents=True, exist_ok=True)
        # Train-image counterparts of each adjacency pair
        seen = set()
        ids = []
        with ADJACENCY.open() as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                tid = r["train_image_id"]
                if tid not in seen:
                    seen.add(tid)
                    ids.append(tid)
    else:
        print(f"unknown mode: {mode}")
        sys.exit(2)

    work = []
    for image_id in ids:
        if image_id not in manifest:
            print(f"WARN: {image_id} not in manifest; skipping")
            continue
        path = manifest[image_id]["local_path"]
        if not Path(path).exists():
            print(f"WARN: {image_id} local path missing: {path}")
            continue
        work.append((image_id, path, str(outdir)))

    print(f"mode={mode} total={len(work)}")

    counts = {"NOMINAL": 0, "DEGRADED": 0, "REFUSED": 0, "UNKNOWN": 0}
    cached = 0
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run_one, w) for w in work]
        for fut in as_completed(futs):
            image_id, status, note = fut.result()
            counts[status] = counts.get(status, 0) + 1
            if note == "cached":
                cached += 1
            done += 1
            if done % 20 == 0:
                print(f"  progress {done}/{len(work)} nom={counts['NOMINAL']} deg={counts['DEGRADED']} ref={counts['REFUSED']} cached={cached}")

    print(f"done mode={mode} counts={counts} cached={cached}")


if __name__ == "__main__":
    main()
