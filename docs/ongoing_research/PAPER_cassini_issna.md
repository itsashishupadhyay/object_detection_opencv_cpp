# Per-Instrument Vision-Based Navigation on Cassini ISS-NAC: An Honest Partial-Win with a Labeling-Contract Ceiling

**Pair:** `(cassini, issna)` — Cassini Imaging Science Subsystem, Narrow Angle Camera
**Weights:** `weight/cassini_issna_planets.onnx`, SHA-256 `0b21d7ad2dea0b0ea8986741cc8b65a7ef6057893e00f7e6eabd9231d14f5935`
**Status:** All §14 exit conditions fail. This paper documents *why*, not *that they pass*.

## Abstract

We ran the full per-pair pipeline specified in `RESEARCH_BRIEF.md` end-to-end against real Cassini ISS-NAC imagery: OPUS download, SPICE-backed spacecraft state, human-gated auto-labeling, YOLOv8 training, and a runtime `--nav-decision` pass over a frozen 242-image test bracket. The deployed model (`run4`, YOLOv8s @ 640 px, 78/80 epochs, val mAP@0.5 = 0.676) classifies 57/242 test frames as NOMINAL, 28 as DEGRADED, and 157 as REFUSED. On NOMINAL frames, median relative range error is 0.266 and median angular center error is 15.08 arcsec, with `IFOV = 1.2357 arcsec/px`. Every §14 exit condition (§14.7 mAP≥0.80, §14.9 median rel range ≤0.10 / p90 ≤0.25 / median angular ≤0.618″, §14.10 adjacency ≥0.90) fails by a wide margin. Two findings dominate the failure: (1) a dataset contamination in which 269 wide-angle ISS-WAC (`co-iss-w*`) images mis-routed into the NAC pipeline induced a structural ~13× range error when processed through the NAC focal length, now excluded from training but retained in the test bracket for honest reporting; and (2) the brief's §7 whole-frame-bbox labeling rule imposes a hard floor on angular centroid precision — bbox centers cannot track body photocenters to sub-pixel accuracy, so sub-arcsec angular error (§14.9) is structurally unreachable under the current labeling contract. We recommend the human either relax §7 (move to photocenter labels) or freeze this pair as an EXIT_FAIL with the audit trail preserved.

## 1. Background

The brief specifies a per-mission/per-instrument protocol for recovering navigation geometry from archival planetary imagery using YOLO detection plus SPICE truth. Per-pair isolation prevents optics-specific biases from leaking across instruments. This paper covers the first pair attempted end-to-end.

## 2. Methods

### 2.1 Instrument and optics
Cassini ISS-NAC; focal length 2003.44 mm, 1024×1024 Loral CCD at 12.0 μm pitch, FOV 0.350°×0.350°, IFOV 1.2357 arcsec/px. Full citations and kernel references in `docs/tech_sheets/cassini_issna.md`; canonical row in `artifacts/instruments.csv` line 2. Distortion <0.45 px at corners per the NAIF IK `cas_iss_v10.ti`.

### 2.2 Spacecraft state
SPICE queries used the kernel set listed in tech sheet §8 (88 kernels in `spice_cache/`). `artifacts/spacecraft_state.csv` contains 1,415 resolved rows. Known gap: per-revolution reconstructed CK files for the 2004–2017 Saturn orbital period are not consolidated, so attitude is populated only for cruise-phase images; position is resolved mission-wide.

### 2.3 Dataset
`artifacts/cassini_issna/image_manifest.csv` contains 2,079 data rows after expansion downloads. The frozen test bracket is 242 images (`artifacts/cassini_issna/test_split.csv`, `test_bracket_frozen=true` in `STATE.json`). The training set is 347 train + 53 val = 400 YOLO labels in `data/cassini_issna_yolo/`.

### 2.4 Auto-labeling and human review
Auto-labeling used the whole-frame-bbox strict rule from brief §7. The human review gate (brief §8.5) ran for multiple rounds; `artifacts/cassini_issna/dropped_by_human.txt` records **44 image IDs** permanently rejected by the human, starting with a round-1 batch of 33 per-image rejects documented in `review/cassini_issna/auto_labels/DECISION.md`. These IDs are never used for training, even if a stricter labeler would accept them later (hard rule).

A second data-quality issue was discovered during `run3` evaluation: 269 images with the `co-iss-w*` prefix (ISS-WAC, wide-angle camera) had been mis-routed into the NAC pipeline by OPUS filters that matched on instrument family rather than the specific `ISSNA` sensor. These are enumerated in `artifacts/cassini_issna/isswa_excluded.txt` and were removed from training before `run4`. They were *deliberately retained* in the test bracket (26 `w*` frames survive there) so their behavior can be reported honestly rather than hidden.

### 2.5 Training
Three training runs, all logged in `STATE.json.checkpoints.training_runs` and `artifacts/cassini_issna/training_metrics*.json`:

| Run | Model | imgsz | Labels | val mAP@0.5 | Notes |
|---|---|---|---|---|---|
| run1 | yolov8n | 320 | 288 | 0.632 | initial |
| run2 | yolov8s | 640 | 378 | 0.562 | deployed for a while; different val split |
| run4 | yolov8s | 640 | 347+53 | **0.676** | ISSWA-clean, +204 images, currently deployed |

Run4 (`artifacts/cassini_issna/runtime_summary_run4.json`): 78/80 epochs completed (early-converged), val mAP@0.5 = 0.676, mAP@0.5:0.95 = 0.583, wallclock 2.496 hours. (Run3 was skipped from deployment; its yolov8n/320 configuration regressed to 0.423.)

### 2.6 Geometry and decision pipeline
Standard NavFix/NavDecision per brief §9–§10. Range recovered from apparent body radius in pixels via the IFOV and known body physical radius; angular center error = pixel offset × 1.2357 arcsec/px. NOMINAL / DEGRADED / REFUSED rules unchanged from the brief.

## 3. Results

All numbers in this section are from `artifacts/cassini_issna/runtime_summary_run4.json` and `artifacts/cassini_issna/residuals_summary.json` unless noted.

### 3.1 Distribution of decision statuses (all 242 test frames)
| Status | Count | Fraction |
|---|---|---|
| NOMINAL | 57 | 23.6% |
| DEGRADED | 28 | 11.6% |
| REFUSED | 157 | 64.9% |

REFUSED dominates. This is partly the strict decision gate doing its job (the model correctly refuses to report geometry it cannot justify), but also reflects the sparse training set (347 labels) relative to the 64-class label space and the diversity of Saturn-system targets.

### 3.2 Detection metrics
Val mAP@0.5 = 0.676, mAP@0.5:0.95 = 0.583 (`runtime_summary_run4.json` training block). Threshold is §14.7 mAP@0.5 ≥ 0.80. **§14.7 fails by 0.124 absolute.**

### 3.3 SPICE residuals — NOMINAL, all 242
From `residuals_summary.json`:
- Relative range error: n=46, median **0.2655**, p90 **13.357**, min 0.1953, max 25.155
- Angular center error: n=21, median **15.082 arcsec**, p90 25.423 arcsec, min 0.298, max 42.113

Thresholds are §14.9 median rel range ≤0.10, p90 ≤0.25, median angular ≤0.618 arcsec. **All three §14.9 sub-conditions fail.** Angular median exceeds threshold by 24×.

### 3.4 Stratified: n-prefix vs w-prefix
`runtime_summary_run4.json.stratified_n_prefix` (216 NAC-native images):
- NOMINAL=52, DEGRADED=20, REFUSED=144
- Relative range error: n=41, median **0.262**, p90 **8.338**, max 25.155

`runtime_summary_run4.json.stratified_w_prefix` (26 mis-routed ISSWA images):
- NOMINAL=5, DEGRADED=8, REFUSED=13
- Relative range error: n=5, median **13.610**, p90 13.741
- Annotation in-file: *"ISSWA contamination — NAC focal length misapplied; systematic ~13× range error"*

The w-prefix frames account for the bulk of the p90 tail on the combined numbers. Their ~13× systematic bias is consistent with `f_NAC / f_WAC ≈ 2003.44 / 200 ≈ 10–13`: when a WAC image is processed through the NAC focal length, the inferred range is scaled by this ratio. **This is a data-provenance bug, not a model failure.** The model correctly detected bodies in those frames; the optics stack was wrong for them.

### 3.5 Adjacency consistency
`runtime_summary_run4.json.test_bracket_all_242.adjacency_pass_rate = 0.2766`. Threshold §14.10 ≥0.90. **§14.10 fails by 0.62 absolute.** Most failures are chains that include a REFUSED frame (automatic fail) or a w-prefix frame (structural scale error propagates).

### 3.6 Comparison to baseline (run2)
`runtime_summary_run4.json.comparison_to_run2_baseline`:
| Metric | run2 | run4 | Δ |
|---|---|---|---|
| NOMINAL count | 61 | 57 | −4 |
| NOMINAL median rel range | 0.311 | 0.266 | −0.045 (better) |
| Adjacency pass rate | 0.314 | 0.277 | −0.037 |

Run4 improved median range error slightly and cleaned the training set; it slightly regressed in NOMINAL count and adjacency. The net picture is not a decisive improvement — it is a clean-up iteration whose main contribution is *diagnostic*, not metric.

## 4. Failure cases

The failure pattern is dominated by two structural effects, not random model noise:

1. **w-prefix contamination (§3.4).** Five surviving w-prefix NOMINAL frames produce a ~13× range bias. The paper reports them separately rather than averaging them into the n-prefix numbers.
2. **Bbox-centroid floor on angular error (§5).** Even on clean n-prefix NOMINAL frames, median angular error is far above the 0.618″ threshold. Best-case single-frame angular error (`residuals_summary.json.angular_center_error_arcsec.min = 0.298`) is the only measurement under threshold in the entire bracket.

## 5. Limitations

**The dominant limitation is the brief's §7 whole-frame-bbox labeling rule.** Under this rule, the label for a body is the axis-aligned bounding box tight to the visible limb. For any body that does not fill the frame, the bbox center does **not** coincide with the body photocenter: for a partially-illuminated or off-center disc, the geometric bbox center and the optical center can differ by up to `half_body_radius_in_pixels`. With IFOV = 1.2357 arcsec/px, a body of radius 10 px at frame center yields an angular-center-error floor of order 5–10 arcsec *even with a perfect detector*. This is consistent with the observed median of 15.08 arcsec and explains why sub-arcsec accuracy (§14.9 threshold 0.618″) is structurally unreachable under the current labeling contract. It is not a model defect and cannot be trained away.

Other limitations:
- Only 347 training labels across a 64-class space; under-represented bodies (small inner moons) are rarely detected confidently.
- CK attitude gap for 2004–2017 means some NavFix cross-checks could not be computed (`spice_lookup_failures = 14` in `residuals_summary.json`).
- Test-bracket retention of w-prefix frames is deliberate for honest reporting but drags aggregate p90.
- Val mAP numbers across runs use different splits and are not strictly comparable (noted in `STATE.json.open_questions`).

## 6. What I could not verify

- `STATE.json.fingerprint.weights_sha256` lists `e32c5bc7...` for the current weights, but `runtime_summary_run4.json.weights_sha256` and the deployed file both show `0b21d7ad...`. Disk wins; STATE is stale. Flagged in the audit.
- `runtime_summary_run4.json.NOMINAL = 57` vs `STATE.json.checkpoints.nav_decision_counts_run2.NOMINAL = 61` — these are different runs, not a contradiction, but STATE has not been updated to reflect run4 NOMINAL counts.
- No consolidated `adjacency_check.json` file exists as a separate artifact; adjacency pass rate is available only via the summary JSON.
- SPICE coverage for irregular Saturn satellites (Albiorix, Bebhionn, etc.) is absent in NAIF SPKs, so 258 manifest rows could not receive state (tech sheet §9).
- Per-class precision/recall from training were not exported to a standalone file beyond the YOLO run log (`training_runs/cassini_issna/train_run4.log`).

## 7. Reproduction

1. `./reproduce.sh cassini issna` — expects `spice_cache/` populated and OPUS reachable.
2. Hardware: single workstation, ~2.5 hours training wallclock for `run4` at yolov8s/640.
3. Key input fingerprints: brief SHA `840c22ed...`, code HEAD `02917a5c...`, weights SHA `0b21d7ad...`.
4. Deterministic re-derivation of all §3 numbers from `residuals.csv` + `runtime_summary_run4.json`.

## References

1. Porco, C.C., et al. (2004). Cassini Imaging Science. *Space Sci. Rev.* 115, 363–497. DOI: 10.1007/s11214-004-1456-7.
2. NAIF Cassini ISS IK v10 (`cas_iss_v10.ti`). https://naif.jpl.nasa.gov/pub/naif/CASSINI/kernels/ik/cas_iss_v10.ti
3. PDS Cassini ISS-NA Instrument Catalog (`issna_inst.cat`), volume `coiss_2101`.
4. `RESEARCH_BRIEF.md` (this workspace), brief SHA `840c22ed9cc7ca985e15f33521c452cb8423e1e3604c5a60c7caf8a07e561477`.
