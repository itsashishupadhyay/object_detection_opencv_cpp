# SELF_AUDIT — (cassini, issna)

**Audit date:** 2026-04-12
**Paper:** `PAPER_cassini_issna.md`
**Result:** **EXIT_FAIL.** Multiple §14 exit conditions fail. Pair is NOT done.

This audit maps every §14 exit condition to a disk-backed verdict. Every number traces to a file path.

---

## §14.1 — instruments.csv has a fully-verified row
**Verdict: PASS**
`artifacts/instruments.csv` line 2 contains the cassini/issna row. All required columns populated (sensor_model, 1024×1024, 12.0 μm, 2003.44 mm, 190.80 mm, 0.350°, 1.2357 arcsec/px, CASSINI_ISS_NAC). `source_url` points to the NAIF IK; `source_citation` includes Porco 2004 + PDS `issna_inst.cat`. `verified_at = 2026-04-08`. Cross-checked against `docs/tech_sheets/cassini_issna.md`.

## §14.2 — tech sheet exists with primary-source citations
**Verdict: PASS**
`docs/tech_sheets/cassini_issna.md` exists, 199 lines, four numbered primary-source references (Porco 2004, PDS catalog, PDS context description, NAIF IK). Every numerical value matches the CSV.

## §14.3 — SPICE kernels cached and spacecraft_state.csv computed
**Verdict: PARTIAL**
`STATE.json.checkpoints.spice_kernels_cached = "88 kernel files in spice_cache/"`. `spacecraft_state.csv` has 1,415 rows (`STATE.json.checkpoints.state_rows_total`). Known and documented gap: tech sheet §8 notes no consolidated reconstructed CK files exist for the 2004–2017 Saturn orbital period, so attitude is populated only for cruise-phase images. Position is resolved mission-wide. `residuals_summary.json.spice_lookup_failures = 14` indicates a small number of runtime SPICE misses.

## §14.4 — image manifest exists with real downloads
**Verdict: PASS**
`artifacts/cassini_issna/image_manifest.csv` = 2,080 lines (1 header + **2,079 data rows**). Expansion download summaries in `expansion_download_summary.json` and `issna_expansion2_summary.json`.

## §14.5 — train/test split frozen, no leakage
**Verdict: PASS**
`STATE.json.checkpoints.train_test_split_done = true`, `test_bracket_frozen = true`, `test_count = 242`. Splits in `train_split.csv` / `test_split.csv`.

## §14.6 — human review gate ran and DECISION.md was respected
**Verdict: PASS (with notes)**
`artifacts/cassini_issna/dropped_by_human.txt` header: *"Permanent human-drop list ... Source: review/cassini_issna/auto_labels/DECISION.md (round 1, 33 per-image rejects)"*. File contains 44 image IDs total. The round-1 batch was 33; the additional 11 come from later review rounds. All are excluded from training permanently per brief §8.5 hard rule. Multiple review rounds occurred (exact count not enumerated in a single file; the paper does not claim a specific round count).

## §14.7 — detection val mAP@0.5 ≥ 0.80
**Verdict: FAIL**
`runtime_summary_run4.json.training.val_best_mAP50 = 0.676`. Threshold 0.80. **Fails by 0.124 absolute.** mAP@0.5:0.95 = 0.583. Deployed weights SHA `0b21d7ad2dea0b0ea8986741cc8b65a7ef6057893e00f7e6eabd9231d14f5935`. Responsible upstream agent: `training` (but see root-cause note below).

## §14.8 — runtime `--nav-decision` executed over test bracket
**Verdict: PASS**
`runtime_summary_run4.json.test_bracket_all_242` populated: NOMINAL=57, DEGRADED=28, REFUSED=157, sum=242. `STATE.json.checkpoints.evaluation_completed = true`. `STATE.json.checkpoints.exit_14_8_met = true`.

## §14.9 — NOMINAL residuals meet thresholds
**Verdict: FAIL (all three sub-conditions)**
From `residuals_summary.json.NOMINAL`:
- Median rel range error = **0.2655** (threshold ≤0.10). **FAIL by 2.66×.**
- p90 rel range error = **13.357** (threshold ≤0.25). **FAIL by 53×.**
- Median angular center error = **15.082 arcsec** (threshold ≤0.618). **FAIL by 24×.**

Stratification (`runtime_summary_run4.json.stratified_*`):
- n-prefix (NAC-native, 216 imgs): NOMINAL=52, median rel range = 0.262, p90 = 8.338.
- w-prefix (ISSWA-contaminated, 26 imgs): NOMINAL=5, median rel range = 13.610, p90 = 13.741.

**Data-quality finding (not a model failure):** The w-prefix contamination is a structural ~13× range bias from applying the NAC focal length (2003.44 mm) to WAC images. 269 w-prefix IDs are listed in `artifacts/cassini_issna/isswa_excluded.txt` and were removed from training before run4; 26 remain in the test bracket for honest reporting and account for the bulk of the combined p90 tail. Responsible upstream agent for the contamination: `data-acquisition` (OPUS filter matched on instrument family, not sensor).

**Root-cause finding for the remaining residuals (even on clean n-prefix data):** The brief's §7 whole-frame-bbox labeling rule imposes a hard floor on angular centroid precision. Bbox center ≠ body photocenter for any body not filling the frame. With IFOV 1.2357 arcsec/px, a 10-px-radius body yields an intrinsic angular-centroid-error floor of order 5–10 arcsec. **The 0.618 arcsec threshold (§14.9) is structurally unreachable under the §7 labeling contract.** This is not fixable by retraining. Responsible decision-maker: the **human** must choose whether to relax §7.

## §14.10 — adjacency pass rate ≥ 0.90
**Verdict: FAIL**
`runtime_summary_run4.json.test_bracket_all_242.adjacency_pass_rate = 0.2766`. Threshold 0.90. **Fails by 0.62 absolute.** Dominant failure modes: chains containing REFUSED frames (65% of bracket is REFUSED) and w-prefix scale errors propagating through chains.

## §14.11 — audit trail complete (iteration logs)
**Verdict: PASS**
`STATE.json.iteration = 14`. All iteration logs in `research_log/` follow the `NNNN_*.md` convention per brief §12. `last_action_at_utc = 2026-04-12T11:00:00Z`.

## §14.12 — reproducibility fingerprint recorded
**Verdict: PARTIAL — stale field**
`STATE.json.fingerprint.brief_sha256 = 840c22ed9cc7ca985e15f33521c452cb8423e1e3604c5a60c7caf8a07e561477` ✅
`STATE.json.fingerprint.code_head_sha = 02917a5c10c6739a857214ecfa621f57c3c037b4` ✅
`STATE.json.fingerprint.weights_sha256.cassini_issna_planets.onnx = e32c5bc7...` ❌ — does **not** match the deployed weight file. `runtime_summary_run4.json.weights_sha256 = 0b21d7ad2dea0b0ea8986741cc8b65a7ef6057893e00f7e6eabd9231d14f5935` is the correct current value. Per brief §0 rule 9 (disk wins), STATE.json needs an orchestrator-side patch. Not a paper-writer job. Flagged for orchestrator.

## §14.13 — paper exists and self-audit passes
**Verdict: FAIL (self-audit does not pass)**
`PAPER_cassini_issna.md` exists. This self-audit exists. But §14.7, §14.9 (×3), and §14.10 all fail. `STATE.json.checkpoints.self_audit_passed = false` is correct.

---

## Summary table

| Condition | Verdict | Delta from threshold |
|---|---|---|
| §14.1 instruments.csv row | PASS | — |
| §14.2 tech sheet | PASS | — |
| §14.3 SPICE + state | PARTIAL | CK gap 2004–2017 (known) |
| §14.4 manifest | PASS | 2,079 rows |
| §14.5 split frozen | PASS | 242 test |
| §14.6 human review | PASS | 44 drops respected |
| §14.7 mAP ≥ 0.80 | **FAIL** | 0.676, short by 0.124 |
| §14.8 runtime ran | PASS | — |
| §14.9a median rel range ≤ 0.10 | **FAIL** | 0.266, 2.66× over |
| §14.9b p90 rel range ≤ 0.25 | **FAIL** | 13.36, 53× over |
| §14.9c median angular ≤ 0.618″ | **FAIL** | 15.08″, 24× over |
| §14.10 adjacency ≥ 0.90 | **FAIL** | 0.277, short by 0.62 |
| §14.11 audit trail | PASS | iter 14 |
| §14.12 fingerprint | PARTIAL | weights SHA stale in STATE |
| §14.13 self-audit passes | **FAIL** | pair not done |

## Recommendation to the human

The pair is **not** closable under the current brief §7 contract. Two realistic paths:

1. **Relax §7 to photocenter labels** (keypoint or ellipse-fit rather than axis-aligned bbox). This unblocks §14.9c angular accuracy, which is currently structurally unreachable. The training pipeline can be reused; only the label-generation and NavFix-centroid stages change. Expected to also improve §14.9a/b because photocenter-driven range recovery is less sensitive to limb phase.
2. **Freeze `(cassini, issna)` as EXIT_FAIL**, preserve the audit trail, and move to the next pair. The diagnostic value of this run (the w-prefix contamination pattern; the bbox-centroid ceiling) transfers to every subsequent pair and is worth preserving even without a closing pass.

The paper-writer has no authority to choose. Control returned to orchestrator for human escalation.

## Upstream agent responsibilities for each failure
- §14.7 (mAP): `training` — but bounded by label quality/quantity; likely needs more labels, which means another `auto-label` round.
- §14.9a/b (range): `training` + `data-acquisition` (w-prefix cleanup in the test bracket is a judgment call).
- §14.9c (angular): **human / brief** — cannot be fixed by any agent under current §7.
- §14.10 (adjacency): downstream of §14.9; fixes when §14.9 fixes.
- §14.12 (stale weights SHA in STATE): **orchestrator** — one-line patch.
