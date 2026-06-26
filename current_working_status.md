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
**Phase −1 — training phase wrapped up (inference-phase prep).** Repo back to clean only-main + pushed.
Full `pytest` **GREEN (267 passed, 1 skipped)** in Docker: fixed the 2 train-smoke MLflow failures at root
cause (MLflow 3.x's `set_tracking_uri` leaks `MLFLOW_TRACKING_URI` into `os.environ`, so in-process
`train.main()` calls cross-contaminated tracking stores → autouse fixture clears it; *not* deferred), and
removed a stale foundation test. Durable artifacts **backed up to `gs://rts-mapping-v2/RTS_MODEL_V2/`**:
runs 110/110 parity (slim — no `resume_latest`), calibration + test_realistic + object_operating_point
(sample restore byte-verified). New SSoT doc `computing/artifact_inventory.md` (artifact→bucket/region map,
cross-linked from `infrastructure.md §4`); `/mnt/outputs/README.md` rewritten to current reality.
`deployment.yaml` min_blob → **2000** (first precision-leaning product — a vectorization-stage param that
does *not* affect the probability COGs). Prior: Phase-D calibration + ensemble deploy decision (ledger H/H.2/J).

<!-- NOW:BEGIN -->
### Now
**Inference phase — building the dual-fleet pan-Arctic pass** (approved plan: 8×A100 master + on-demand
4× g2-standard-96 L4, auto-balancing GCS shard-claim queue; writes Float32 probability COGs to
`gs://rts-mapping-v2-usw1/inference/2025q3_south/`, threshold/min_blob applied later at vectorization).
**Long pole = the 2025_south Sentinel-2 export** (NDVI source): GEE-throttled (task queue full, backing off),
~51% of 1799 cells, days out — gates launch; everything else builds in parallel. **Buildable now (no S2 dep):**
Phase 1 shard-claim queue (`scripts/shard_tiles.py`, `inference/claim.py`, `scripts/run_inference_worker.py`
+ refactor `inference.py` body into `run_inference(...)`, + tests), Phase 2 image rebuild `rts-infer:v1`
(registry `:v2` has MLflow 3.12 + stale pre-Phase-D code) + 3 per-seed deployment packages + output-bucket
layout, and the progress monitor/watcher (`scripts/inference_progress.py`). **Master = `a100-8x-train`,
never stop/rename (A100 scarcity).** Compute is shared PDG project — only ever touch our `rts-`/`rts-infer-*`
resources.
<!-- NOW:END -->

### Future plans (inference phase — full plan in `.claude/plans/elegant-exploring-lemur.md`)
1. **Phase 0** — finish the 2025_south S2 export (GEE-throttled), build + upload the `s2_index` (NDVI
   windowing) + domain↔S2 coverage audit with a residual-gap policy; build `scripts/inference_progress.py`
   (terminal dashboard + Claude-watcher JSON).
2. **Phase 1** — GCS shard-claim queue: `scripts/shard_tiles.py` (spatial-contiguous shards), `inference/
   claim.py` (atomic claim/done/stale-reclaim), `scripts/run_inference_worker.py`; refactor `inference.py`
   body into `run_inference(tiles_df, …)`; tests (atomic mutual-exclusion, reclaim, done-skip, exactly-once).
3. **Phase 2** — rebuild `rts-infer:v1` (current code + cv2 baked + MLflow `<3.0` per requirements; push via
   ADC); build the 3 per-seed deployment packages; create the output-bucket layout (hierarchical shard-scoped
   prefixes, one manifest/shard).
4. **Phase 3** — fleet provisioning (`create_inference_fleet.sh`, `rts-infer-{1..4}`) + pre-flight: L4 quota,
   1-VM startup test, drift check, Banks Island RGB+NDVI end-to-end + multi-VM claim collision check +
   kill/restart drill, benchmark → shard size + output dtype + ETA.
5. **Phase 4** — launch 40 workers (explicit go), monitor + auto-stop watchdog (stops `rts-infer-*` only;
   never the A100 master), end-of-run exactly-once coverage reconciliation; then stop (not delete) the L4 fleet.
6. **Deferred (per user)**: post-inference vectorization → products (cut from retained prob COGs, no GPU rerun);
   North S2-RGB model; v3 backlog (re-stage, MAE SSL, multi-scale — see ledger "Deferred to v3").

---

## Pointers

- **Experiments / recipe / findings** → `docs/experiment_ledger.md` (SSoT)
- **Visual report** → `docs/report.html` (generated from the ledger)
- **Optimization opportunities + fairness audit** → `docs/optimization_roadmap.md`
- **Inference pipeline** → `inference/inference.md` · **infra/budget** → `computing/infrastructure.md`
- **Pre-2026-06-25 dated status / decisions / dev-log** → `docs/archive/working_status_pre-rolling_2026-06-25.md`
