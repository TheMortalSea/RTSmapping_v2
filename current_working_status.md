# Working Status — RTSmapping_v2

Project diary: where we are and where we're going. **Rolling** — the *Now* section is overwritten each
update and the *Just completed* step rolls forward; old detail is not accumulated here (history lives in
git and `docs/archive/`). Experiment numbers + the locked recipe are **not** duplicated here — they live
in `docs/experiment_ledger.md` (the SSoT); this doc links to it. Update ritual: see `CLAUDE.md`.

---

## Project Summary

Semantic segmentation of **Retrogressive Thaw Slumps (RTS)** in Arctic satellite imagery (60–74°N).
Train on 2024 PlanetScope Quarterly Basemap (RGB, ~3 m), deploy inference on 2025 imagery for a
pan-arctic RTS survey map. Solo research project — flat code, minimal abstraction.

**Core constraints** (non-negotiable, see `CLAUDE.md`): CRS EPSG:3857 · tile 512×512 · labels
0=bg/1=RTS/255=ignore · per-dataset z-score norm (`normalization_stats.json`) · seed 42, deterministic.

**Stack**: PyTorch 2.x + `segmentation_models_pytorch` (UNet++/EffB5), albumentations, rasterio,
geopandas, MLflow. Compute: 8× A100-80GB (`a100-8x-train`) for training; 32× L4 (us-west1) provisioned
for inference. Docker `rts-train:v2`. Data in `gs://abrupt_thaw/` + `gs://rts-mapping-v2*`.

**Dataset**: v1.0 standard (22,259 tiles; 1,718 pos / 20,541 neg), corrected leakage-free region split.
**Diagnosis**: representation-limited, not data-volume- or capacity-limited (see ledger family B).

---

## Rolling progress

### Just completed
**Fair encoder A/B settled → EffB5.** The sat-DINOv3 + NDVI re-run on the *full locked recipe* (the only
clean comparison) tied EffB5 across PR-AUC + object metrics, collapsing the earlier confounded "sat edge"
→ EffB5 is the v2 encoder (cheaper, equal). This closes the modeling campaign (every planned screen run;
recipe + per-family verdicts in `docs/experiment_ledger.md`). Earlier: repo consolidated to only-`main`;
the three living docs rebuilt around one SSoT + score-harvest (see `CLAUDE.md`).

<!-- NOW:BEGIN -->
### Now
**Encoder DECIDED = EffB5** — the modeling campaign is fully closed. The fair sat-DINOv3 + NDVI re-run on
the locked recipe finished (3-seed `best_smoothed` 0.9221/0.9286/0.9067 = **0.9191**) and **ties** EffB5
(**0.9218**) on PR-AUC *and* object metrics (IoU 0.612, obj-F1 ≈0.44 both) — the earlier sat "edge" was a
boundary-treatment confound that vanished on the matched recipe. So the v2 deploy encoder is **EffB5**
(= `aug_trivialaugment_deploy`, 3 clean checkpoints, 0.9218), ~4× cheaper than the ViT-L at no accuracy
cost; ensemble dropped (sat adds nothing). Immediate next: **Phase D** — H calibration (temperature +
threshold + D4-TTA) on Val-Realistic → 3-seed final → **Test-Realistic once** → package.
<!-- NOW:END -->

### Future plans
1. **Phase D — calibrate EffB5 + lock** (family H/I): temperature + threshold + D4-TTA on Val-Realistic on
   the EffB5 recipe (`aug_trivialaugment_deploy`, 3-seed 0.9218) → **Test-Realistic once** → package.
   (Encoder already decided = EffB5; sat-DINOv3 tied on the fair re-run, ensemble dropped.)
3. **Inference build-out**: NDVI-at-inference reader (built); quad-level LRU cache + spatial hit-test
   (`docs/optimization_roadmap.md`); output bucket; pre-flight on the 32× L4 fleet → full pan-arctic pass
   (309,100 quads → 41.57M tiles, 20.68M km²).
4. **Post-inference**: vectorize, QC, hard-negative mining (feeds v3).
5. **v3 backlog** (deferred): v1.0 re-stage (+28 pos / −49 black), MAE SSL pretraining, context-expansion
   multi-scale. See ledger "Deferred to v3".

---

## Pointers

- **Experiments / recipe / findings** → `docs/experiment_ledger.md` (SSoT)
- **Visual report** → `docs/report.html` (generated from the ledger)
- **Optimization opportunities + fairness audit** → `docs/optimization_roadmap.md`
- **Inference pipeline** → `inference/inference.md` · **infra/budget** → `computing/infrastructure.md`
- **Pre-2026-06-25 dated status / decisions / dev-log** → `docs/archive/working_status_pre-rolling_2026-06-25.md`
