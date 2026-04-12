# supporting_scripts/ — the Python + shell pipeline around the C++ binary

The C++ binary in this repository does one job well: **given an image and an ONNX model, emit a NavDecision JSON.** Everything else — pulling images off OPUS, computing SPICE ground truth, auto-labeling, splitting, batch-running, scoring residuals, and the adjacency consistency check — lives here as small, single-purpose Python scripts.

This doc walks through every script in the order you'd actually call them to reproduce the Cassini ISS-NAC run end-to-end. Each section lists: what the script does, what it reads, what it writes, and how to invoke it.

Most scripts expect to find a virtualenv at `../.yolo_venv` (created once with `python3 -m venv .yolo_venv && pip install -r requirements.txt` from the repo root). SPICE-touching scripts add that venv's `site-packages` to `sys.path` explicitly — no `source activate` needed.

> **Absolute paths:** several of these scripts currently hard-code `/Users/upadhyay/dev/ICES/object_detection_opencv_cpp` as the workspace root. If you clone this repo elsewhere, grep for `WORKSPACE` / `WS =` at the top of each file and adjust. A future cleanup pass will replace those with `os.path.dirname(os.path.abspath(__file__))`.

---

## 0. Pipeline map

```
   ┌────────────────────────┐
   │ opus_stratified_       │  ① Download raw ISS-NAC images from OPUS,
   │ downloader.py          │     stratified by body, hashed, manifested.
   └────────────┬───────────┘
                │ image_manifest.csv + data/cassini/issna/<body>/images/
                ▼
   ┌────────────────────────┐
   │ compute_spacecraft_    │  ② Query SPICE for spacecraft state at every
   │ state.py               │     image timestamp. Emits spacecraft_state.csv.
   └────────────┬───────────┘
                │ spacecraft_state.csv
                ▼
   ┌────────────────────────┐
   │ auto_labeler.py        │  ③ Predict body pixel size from SPICE geometry,
   │ (+ round 3 / round 4)  │     refine with OpenCV, sanity-check, write YOLO
   └────────────┬───────────┘     labels + human-review bundle.
                │  labels/*.txt   ★ HUMAN GATE: write DECISION.md ★
                ▼
   ┌────────────────────────┐
   │ make_split.py          │  ④ Stratified train/test split with burst
   └────────────┬───────────┘     grouping + adjacency_pairs.csv.
                │ test_split.csv, adjacency_pairs.csv
                ▼
   ┌────────────────────────┐
   │ (ultralytics train,    │  ⑤ Train the YOLOv8s model, export to ONNX.
   │  then export_yolo26_   │     Not in this directory — runs inside the
   │  opencv.py for yolo26) │     training sub-agent.
   └────────────┬───────────┘
                │ cassini_issna_planets.onnx
                ▼
   ┌────────────────────────┐
   │ run_nav_decision_      │  ⑥ Batch-run the C++ --nav-decision binary
   │ batch.py (or .sh)      │     over every test-bracket image.
   └────────────┬───────────┘
                │ artifacts/cassini_issna/decisions/*.json
                ▼
   ┌────────────────────────┐  ┌────────────────────────┐
   │ compute_residuals.py   │  │ adjacency_check.py     │  ⑦ Score the run
   └────────────┬───────────┘  └────────────┬───────────┘     against SPICE
                │                           │                    truth.
                ▼                           ▼
        residuals.csv              adjacency_check.csv
        residuals_summary.json     adjacency_check_summary.json
```

---

## 1. Image acquisition (OPUS → local disk)

### `opus_image_downloader.py` — interactive OPUS browser
A human-friendly REPL for poking at [NASA's OPUS](https://opus.pds-rings.seti.org/) catalog. Useful for exploring what's available before committing to a stratified download. Prints targets, lets you pick a mission/instrument, downloads a handful of images into `./opus_dataset/`. **Not used in the reproducible pipeline** — kept for discovery.

```bash
python3 supporting_scripts/opus_image_downloader.py
```

### `opus_dataset_downloader.sh` — legacy bash downloader
Older bash-only downloader from an earlier iteration of this project. Kept for historical reference; superseded by the stratified Python downloader below.

### `opus_stratified_downloader.py` — **the main acquisition tool**
Non-interactive, scriptable, resumable. Implements everything `RESEARCH_BRIEF.md §7.1` requires:

- **Stratified sampling by target body** across the whole Cassini timeline so no single Saturn-approach month dominates.
- **Burst preservation** — if OPUS returns ≥3 frames within 60 s, keep the whole burst so adjacency pairs can form.
- **Pre-flight disk-budget check** — refuses to start if the projected download would exceed `--budget-gb`.
- **SHA-256 hashing** of every byte fetched; hashes land in `image_manifest.csv`.
- **Hierarchical layout** — `data/<mission>/<instrument>/<body>/images/<id>.png`.
- **Resume support** — on re-run, skips IDs already in the manifest and cleans up any `.part` files from a previous crash.

Invoke:

```bash
python3 supporting_scripts/opus_stratified_downloader.py \
    --mission cassini --instrument issna \
    --target-count 1200 \
    --output-dir data/cassini/issna \
    --manifest-path artifacts/cassini_issna/image_manifest.csv \
    --budget-gb 20 \
    --batch-size 50
```

Writes: `data/cassini/issna/<body>/images/*.png`, `data/cassini/issna/<body>/metadata/*.json`, and appends to `image_manifest.csv`.

### `opus_expansion_downloader.py` — additive expansion (round 3)
After the initial 1200-image run, we expanded to ~2000 images. This variant loads an exclusion set (existing manifest IDs + the frozen `test_split.csv` + any `dropped_by_human.txt`), pre-filters OPUS candidates **before** stratified sampling, and **appends** new rows to the manifest without touching existing ones. Refuses to re-download or re-hash anything already in the manifest.

### `opus_issna_only_expansion.py` — ISSNA-only second expansion (round 4)
Final expansion for the ISS-NAC pair. Hard-excludes every `co-iss-w*` (ISS Wide Angle) ID at every stage — we had discovered that ISSWA images get mis-labeled with ISSNA focal length by the tech sheet, inflating range errors ~13×. This script only accepts `co-iss-n*` prefixes.

---

## 2. Ground truth from SPICE kernels

### `compute_spacecraft_state.py` — **SPICE → spacecraft_state.csv**
For every image in the manifest, queries SPICE kernels (LSK/SCLK/FK/IK/PCK/SPK/CK — see `docs/tech_sheets/cassini_issna.md §8`) to compute:

- Spacecraft position + velocity **relative to the target body** in J2000.
- Spacecraft attitude quaternion from the CK kernels.
- Target body direction from the spacecraft (unit vector in the spacecraft frame, used later as residuals ground truth).
- Sun and Earth directions (for illumination geometry).

Writes one row per successfully resolved image to `artifacts/cassini_issna/spacecraft_state.csv`, and flips `state_verified = true` on the matching manifest row. Images that SPICE can't resolve (kernel gaps, unknown bodies) stay `false` and are skipped downstream.

```bash
python3 supporting_scripts/compute_spacecraft_state.py
```

### `compute_spacecraft_state_expansion.py` — incremental version
Same logic but only processes manifest rows that don't already have a state row. **Appends** to `spacecraft_state.csv` without rewriting existing rows. Run after each expansion download.

---

## 3. Auto-labeling (with a mandatory human gate)

### `auto_labeler.py` — **the risky step**
For each image with verified SPICE state, this script:

1. **Predicts** the body's pixel footprint from SPICE geometry:
   `pixel_radius = body_radius / range * (focal_length / pixel_pitch)`
2. **Refines** the predicted box by running OpenCV contour detection inside a padded ROI around the SPICE-predicted center.
3. **Sanity-checks** each candidate: bbox must be >5 px, <95% of frame, contour area must agree with SPICE prediction within tolerance, no NaN/inf, class must be in the allowed 64-body list.
4. **Writes** YOLO-format labels to `data/cassini/issna/<body>/labels/<image_id>.txt` — one line, `class cx cy w h` normalized to `[0, 1]`.
5. **Renders a human review bundle** to `review/cassini_issna/auto_labels/` with side-by-side image + overlay thumbnails and a `REVIEW_REQUEST.md` pointing at each proposed label.

Outputs `class_names.txt` (the 64-class list) and `artifacts/cassini_issna/auto_label_stats.json` (per-body accept/reject counts).

**Per `RESEARCH_BRIEF §8.5`, the pipeline pauses here and waits for the human.** You review the bundle, then write `STATUS: APPROVED | NEEDS_FIXES | REJECTED` into `review/cassini_issna/auto_labels/DECISION.md`. The training sub-agent refuses to run without a non-empty approved `STATUS:` line. The auto-labeler can never approve its own output.

```bash
python3 supporting_scripts/auto_labeler.py
```

### `auto_labeler_expansion.py` — round 3 (additive)
Same thresholds and sanity checks, but **only labels new images** from the round-3 expansion and **preserves** all existing approved round-2 labels byte-for-byte. Writes a fresh review bundle under `review/cassini_issna/auto_labels_round3/`.

### `auto_labeler_round4.py` — round 4 (ISSNA-only)
Same idea, round-4 expansion. Adds the hard `co-iss-w*` exclusion at every stage. Review bundle goes to `review/cassini_issna/auto_labels_round4/`.

---

## 4. Split + adjacency pairs

### `make_split.py` — **stratified train/test split with burst grouping**
Implements `RESEARCH_BRIEF §7.2`:

- Every body with ≥50 images appears in **both** train and test (stratified).
- **Burst grouping** — any two images taken within 60 s of each other land in the same bracket so no "near-duplicate" leaks across the split.
- Target ratio ~80/20, adjusted upward for stratification constraints.
- Images classified as `unknown` are excluded entirely.
- **Fixed random seed** — the split is byte-for-byte reproducible.

Also produces `adjacency_pairs.csv`: for each test image, the nearest train image in time with the **same instrument and filter**. This drives the adjacency consistency check (§12 / exit condition §14.10).

```bash
python3 supporting_scripts/make_split.py
```

Writes: `artifacts/cassini_issna/test_split.csv`, `train_split.csv`, `adjacency_pairs.csv`.

---

## 5. Training export helpers (YOLO26-specific)

### `export_yolo26_opencv.py` — export YOLO26 with the OpenCV-DNN-friendly head
YOLO26 ships with two detection heads by default. OpenCV DNN only understands the "one-to-many" head (`[batch, 84, 8400]`, requires NMS post-processing). This script forces that export option so the resulting ONNX loads cleanly via `cv::dnn::readNetFromONNX`.

```bash
python3 supporting_scripts/export_yolo26_opencv.py --model yolo26n.pt --output yolo26n_opencv.onnx
```

### `quick_export_yolo26.sh` — one-shot wrapper
Downloads the ultralytics weight (if missing), exports it, copies it into `weight/`, and runs a quick smoke test. Convenience only — not part of the Cassini pipeline.

**Note:** the Cassini ISS-NAC model (`weight/cassini_issna_planets.onnx`) is a **YOLOv8s**, not YOLO26 — exported via the standard `ultralytics` CLI inside the training sub-agent. These YOLO26 helpers are kept for the upstream detection framework.

---

## 6. Running the model over the test bracket

### `run_nav_decision_batch.py` — **Python batch runner (preferred)**
For each row in `test_split.csv`, invokes `build/opencv_cpp_release --nav-decision` on the image and captures stdout for status parsing. The C++ binary itself writes each per-image JSON into `artifacts/cassini_issna/decisions/`. This Python wrapper handles:

- Progress logging to `nav_decision_batch.log` (one line per image).
- stderr capture to `nav_decision_batch.stderr.log`.
- Thread-pool parallelism (OpenCV's DNN backend is single-threaded per inference, but running multiple inferences in parallel saturates cores).
- Summary line parsing: `NOMINAL: N | DEGRADED: N | REFUSED: N`.

```bash
python3 supporting_scripts/run_nav_decision_batch.py
```

### `run_nav_decision_batch.sh` — pure-bash equivalent
No threadpool, no Python dependency. Slower but useful when you want to watch the log scroll in real time or when the venv is broken. Same inputs, same outputs.

---

## 7. Scoring the run against SPICE truth

### `compute_residuals.py` — **SPICE residuals per decision**
For every `decisions/*.json` with status `NOMINAL` or `DEGRADED`, looks up the true spacecraft→target-body range and direction via SPICE and computes:

- `relative_range_error = |predicted_range − true_range| / true_range`
- `angular_center_error_arcsec` = angle between the predicted line-of-sight and the true body direction.

`REFUSED` decisions are written to the CSV with `null` numeric fields so the paper can honestly report all 242 rows, but **statistics are stratified** — NOMINAL, DEGRADED, and REFUSED are never aggregated. Honest reporting over nice-looking aggregate numbers.

```bash
python3 supporting_scripts/compute_residuals.py
```

Writes: `artifacts/cassini_issna/residuals.csv` + `residuals_summary.json`.

### `adjacency_check.py` — **exit condition §14.10**
For each `(test_id, train_id)` pair in `adjacency_pairs.csv`:

1. Load the test image's NavDecision JSON.
2. If it's NOMINAL, query SPICE for the spacecraft→body true range at **both** timestamps.
3. The pair **passes** if (a) the predicted body matches the pair's body column and (b) the test-image predicted range is within 25% of the true range at the train timestamp (same tolerance as §14.9).
4. Non-NOMINAL test decisions are excluded from the pass-rate denominator — §14.10 is a NOMINAL-only consistency check.

```bash
python3 supporting_scripts/adjacency_check.py
```

Writes: `artifacts/cassini_issna/adjacency_check.csv` + `_summary.json`.

---

## Order of execution — full reproduction

```bash
# One-time setup
python3 -m venv .yolo_venv
.yolo_venv/bin/pip install -r requirements.txt
# (and install SPICE kernels into spice_cache/ — see docs/tech_sheets/cassini_issna.md §8)

# Pipeline
python3 supporting_scripts/opus_stratified_downloader.py ...   # ①
python3 supporting_scripts/compute_spacecraft_state.py         # ②
python3 supporting_scripts/auto_labeler.py                     # ③
#  → HUMAN: review review/cassini_issna/auto_labels/ and write DECISION.md
python3 supporting_scripts/make_split.py                       # ④
#  → train via ultralytics CLI, export ONNX                    # ⑤
python3 supporting_scripts/run_nav_decision_batch.py           # ⑥
python3 supporting_scripts/compute_residuals.py                # ⑦
python3 supporting_scripts/adjacency_check.py                  # ⑦
```

Everything after step ③ is deterministic given the same inputs. Step ③ itself is deterministic *modulo the human's review choices*, which is the whole point of the review gate.
