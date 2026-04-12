#!/usr/bin/env python3
"""Batch-run the nav-decision C++ binary over all test-bracket images.

Invokes build/opencv_cpp_release --nav-decision for each image in test_split.csv,
parses the per-image status from stdout, and writes a progress log.
The binary writes the per-image JSON to artifacts/cassini_issna/decisions/ itself.
"""
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

WS = "/Users/upadhyay/dev/ICES/object_detection_opencv_cpp"
BIN = os.path.join(WS, "build/opencv_cpp_release")
MODEL = "weight/cassini_issna_planets.onnx"
NAMES = "weight/cassini_issna_planets.names"
TEST_SPLIT = os.path.join(WS, "artifacts/cassini_issna/test_split.csv")
MANIFEST = os.path.join(WS, "artifacts/cassini_issna/image_manifest.csv")
LOG = os.path.join(WS, "artifacts/cassini_issna/nav_decision_batch.log")
STDERR_LOG = os.path.join(WS, "artifacts/cassini_issna/nav_decision_batch.stderr.log")

SUMMARY_RE = re.compile(r'NOMINAL:\s*(\d+)\s*\|\s*DEGRADED:\s*(\d+)\s*\|\s*REFUSED:\s*(\d+)')


def load_manifest_paths(path):
    out = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row['image_id']] = row['local_path']
    return out


def load_test_ids(path):
    out = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row['image_id'])
    return out


def run_one(image_id, abs_path, stderr_fh):
    rel_path = abs_path
    if abs_path.startswith(WS + '/'):
        rel_path = abs_path[len(WS) + 1:]
    if not os.path.isfile(os.path.join(WS, rel_path)):
        if os.path.isfile(abs_path):
            rel_path = abs_path
        else:
            return image_id, 'MISSING_FILE', ''
    try:
        proc = subprocess.run(
            [BIN, '--nav-decision', '--mission', 'cassini', '--instrument', 'issna',
             '-p', rel_path, '-m', MODEL, '-l', NAMES],
            cwd=WS, capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        return image_id, 'TIMEOUT', ''
    except Exception as e:
        return image_id, f'EXCEPTION:{e}', ''
    if proc.stderr:
        stderr_fh.write(f'=== {image_id} ===\n{proc.stderr}\n')
    out = proc.stdout
    m = SUMMARY_RE.search(out)
    if not m:
        return image_id, f'UNKNOWN(exit={proc.returncode})', out[-200:]
    n, d, r = map(int, m.groups())
    if n == 1:
        return image_id, 'NOMINAL', ''
    if d == 1:
        return image_id, 'DEGRADED', ''
    if r == 1:
        return image_id, 'REFUSED', ''
    return image_id, f'UNKNOWN_COUNTS({n},{d},{r})', ''


def main():
    paths = load_manifest_paths(MANIFEST)
    test_ids = load_test_ids(TEST_SPLIT)
    print(f'Loaded {len(paths)} manifest rows, {len(test_ids)} test ids')

    results = {}
    t0 = time.time()
    processed = 0
    missing = 0
    n_count = d_count = r_count = err_count = 0

    # Run in parallel — 4 workers is a reasonable default for CPU inference
    with open(LOG, 'w') as log_fh, open(STDERR_LOG, 'w') as stderr_fh:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = {}
            for image_id in test_ids:
                if image_id not in paths:
                    log_fh.write(f'MISSING_MANIFEST {image_id}\n')
                    missing += 1
                    continue
                fut = ex.submit(run_one, image_id, paths[image_id], stderr_fh)
                futures[fut] = image_id
            for fut in as_completed(futures):
                image_id = futures[fut]
                try:
                    iid, status, tail = fut.result()
                except Exception as e:
                    log_fh.write(f'EXCEPTION {image_id} {e}\n')
                    err_count += 1
                    continue
                results[iid] = status
                if status == 'NOMINAL':
                    n_count += 1
                elif status == 'DEGRADED':
                    d_count += 1
                elif status == 'REFUSED':
                    r_count += 1
                else:
                    err_count += 1
                    log_fh.write(f'NONSTD {image_id} {status} {tail}\n')
                processed += 1
                if processed % 20 == 0:
                    elapsed = time.time() - t0
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f'PROGRESS {processed}/{len(test_ids)}  '
                          f'N={n_count} D={d_count} R={r_count} err={err_count}  '
                          f'{rate:.1f}/s  elapsed={elapsed:.1f}s', flush=True)

    elapsed = time.time() - t0
    print(f'DONE total={len(test_ids)} processed={processed} '
          f'NOMINAL={n_count} DEGRADED={d_count} REFUSED={r_count} '
          f'missing={missing} errors={err_count}  elapsed={elapsed:.1f}s')
    # Emit a JSON side-file for downstream scripts
    import json
    with open(os.path.join(WS, 'artifacts/cassini_issna/nav_decision_batch_summary.json'), 'w') as f:
        json.dump({
            'total': len(test_ids),
            'processed': processed,
            'NOMINAL': n_count,
            'DEGRADED': d_count,
            'REFUSED': r_count,
            'missing': missing,
            'errors': err_count,
            'elapsed_s': elapsed,
            'per_image': results,
        }, f, indent=2)
    return 0 if err_count == 0 and missing == 0 else 0  # non-fatal


if __name__ == '__main__':
    sys.exit(main())
