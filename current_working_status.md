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
**Phase D calibration done → deploy a 3-seed EffB5 ensemble.** Built+tested `scripts/calibrate.py`
(TTA-select → temperature → threshold on Val-Realistic, reusing the deploy-path `predict_probs` for parity);
calibrated all 3 EffB5 seeds. TTA → **none** (D4 hurt, hflip sub-gate); **T≈0.51–0.54** (focal model is
under-confident → calibration sharpens). 3-seed mean-prob ensemble PR-AUC-geomean **0.9393** (vs best single
0.9302), P=0.800/R=0.896. **Decision: ensemble** — for robustness against a single unlucky seed (seed42 0.916
vs seed44 0.930), not the marginal gain. Calibrated values written to `configs/deployment.yaml`. Prior step:
fair encoder A/B settled → EffB5 (sat-DINOv3 tied, dropped).

<!-- NOW:BEGIN -->
### Now
**Phase D — calibrated, ensemble selected; building the ensemble deploy/eval path next.** Calibration on
Val-Realistic is complete (`/mnt/outputs/v1.0/calibration/effb5_trivialaug/calibration_report.json`): deploy =
**3-seed EffB5 ensemble** (mean-prob fusion, **T=0.5123, thr=0.1224, tta=none**), Val PR-AUC-geomean **0.9393**,
P=0.800/R=0.896. Caveat: threshold selected at 1:20 prevalence (val pool limit); realized precision at 1:200–1000
deployment prevalence will be lower — Test-Realistic gives the honest number. **Immediate next:** add 3-model
ensemble support to the deploy path (`package_model.py` per-seed packages + a fusion manifest, `predictor.py`
multi-model load + fuse, `evaluate_test.py` ensemble) → then **Test-Realistic ONCE** (held for explicit go) → package.
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
