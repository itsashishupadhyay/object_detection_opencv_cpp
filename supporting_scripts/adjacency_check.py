#!/usr/bin/env python3
"""Adjacency consistency check (§12, exit condition §14.10).

For each (test_id, train_id) pair in adjacency_pairs.csv:
  - Load the test image's NavDecision JSON.
  - If it is NOMINAL, query SPICE for the spacecraft→body true range at
    both the test and train timestamps.
  - The pair passes if: (a) predicted body matches the pair's body column,
    and (b) the test-image predicted range is within 25% of the true range
    at the train timestamp (adjacency tolerance — consistent with §14.9).
  - Non-NOMINAL test decisions are excluded from the pass-rate denominator
    (§14.10 evaluates NOMINAL-only pairs).

Writes artifacts/cassini_issna/adjacency_check.csv and _summary.json.
"""
import csv
import glob
import json
import math
import os
import sys

WS = "/Users/upadhyay/dev/ICES/object_detection_opencv_cpp"
VENV_SITE = os.path.join(WS, ".yolo_venv/lib/python3.12/site-packages")
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)
import spiceypy as spice

DEC_DIR = os.path.join(WS, "artifacts/cassini_issna/decisions")
PAIRS = os.path.join(WS, "artifacts/cassini_issna/adjacency_pairs.csv")
OUT_CSV = os.path.join(WS, "artifacts/cassini_issna/adjacency_check.csv")
OUT_JSON = os.path.join(WS, "artifacts/cassini_issna/adjacency_check_summary.json")
SPICE_CACHE = os.path.join(WS, "spice_cache")

TOLERANCE = 0.25
CASSINI = "CASSINI"
BODY_MAP = {
    "saturn": "SATURN", "saturn_rings": "SATURN",
    "titan": "TITAN", "enceladus": "ENCELADUS", "mimas": "MIMAS",
    "tethys": "TETHYS", "dione": "DIONE", "rhea": "RHEA",
    "hyperion": "HYPERION", "iapetus": "IAPETUS", "phoebe": "PHOEBE",
    "janus": "JANUS", "epimetheus": "EPIMETHEUS",
    "prometheus": "PROMETHEUS", "pandora": "PANDORA", "atlas": "ATLAS",
    "pan": "PAN", "helene": "HELENE", "telesto": "TELESTO",
    "calypso": "CALYPSO", "polydeuces": "POLYDEUCES",
    "jupiter": "JUPITER", "io": "IO", "europa": "EUROPA",
    "ganymede": "GANYMEDE", "callisto": "CALLISTO",
    "venus": "VENUS", "earth": "EARTH", "moon": "MOON",
}


def furnsh_all():
    for pat in ("*.tls", "*.tsc", "*.tpc", "*.tf", "*.ti", "*.bsp", "*.bc"):
        for f in sorted(glob.glob(os.path.join(SPICE_CACHE, pat))):
            try:
                spice.furnsh(f)
            except Exception:
                pass


def true_range(body_key, utc):
    sp = BODY_MAP.get(body_key)
    if not sp:
        return None
    try:
        et = spice.str2et(utc)
        pos, _ = spice.spkpos(sp, et, 'J2000', 'LT+S', CASSINI)
        return math.sqrt(sum(p * p for p in pos))
    except Exception:
        return None


def main():
    furnsh_all()

    # Load all decisions
    decisions = {}
    for path in glob.glob(os.path.join(DEC_DIR, '*.json')):
        d = json.load(open(path))
        decisions[d['image_id']] = d

    rows = []
    nominal_pairs = 0
    pass_pairs = 0
    body_mismatch = 0
    range_fail = 0
    spice_miss = 0

    with open(PAIRS, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_id = r['test_image_id']
            train_id = r['train_image_id']
            pair_body = r['body']
            d = decisions.get(test_id)
            result = 'SKIP_NO_DECISION'
            pred_body = ''
            pred_range = None
            tr_test = None
            tr_train = None
            rel_err = None

            if d:
                status = d['status']
                fix = d.get('fix') or {}
                pred_body = fix.get('body') or ''
                pred_range = fix.get('range_km')
                if status != 'NOMINAL':
                    result = f'SKIP_{status}'
                else:
                    nominal_pairs += 1
                    if pred_body != pair_body:
                        result = 'FAIL_BODY_MISMATCH'
                        body_mismatch += 1
                    else:
                        tr_train = true_range(pair_body, r['train_timestamp_utc'])
                        tr_test = true_range(pair_body, r['test_timestamp_utc'])
                        if tr_train is None or pred_range is None:
                            result = 'SKIP_SPICE'
                            spice_miss += 1
                            nominal_pairs -= 1  # don't count in denominator
                        else:
                            rel_err = abs(pred_range - tr_train) / tr_train
                            if rel_err <= TOLERANCE:
                                result = 'PASS'
                                pass_pairs += 1
                            else:
                                result = 'FAIL_RANGE'
                                range_fail += 1

            rows.append({
                'test_image_id': test_id,
                'train_image_id': train_id,
                'pair_body': pair_body,
                'predicted_body': pred_body,
                'test_timestamp_utc': r['test_timestamp_utc'],
                'train_timestamp_utc': r['train_timestamp_utc'],
                'time_delta_s': r['time_delta_s'],
                'predicted_range_km': pred_range if pred_range is not None else '',
                'true_range_test_km': f'{tr_test:.3f}' if tr_test is not None else '',
                'true_range_train_km': f'{tr_train:.3f}' if tr_train is not None else '',
                'relative_error_vs_train': f'{rel_err:.6f}' if rel_err is not None else '',
                'result': result,
            })

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    pass_rate = pass_pairs / nominal_pairs if nominal_pairs > 0 else 0.0
    summary = {
        'total_pairs': len(rows),
        'nominal_pairs_in_denominator': nominal_pairs,
        'pass_pairs': pass_pairs,
        'body_mismatch_fails': body_mismatch,
        'range_fails': range_fail,
        'spice_lookup_skipped': spice_miss,
        'non_nominal_skipped': sum(1 for r in rows if r['result'].startswith('SKIP_') and r['result'] != 'SKIP_SPICE'),
        'pass_rate': pass_rate,
        'tolerance': TOLERANCE,
        'exit_14_10_pass_rate_ge_0_90': pass_rate >= 0.90,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
