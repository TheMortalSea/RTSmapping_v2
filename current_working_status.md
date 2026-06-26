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
**Phases 1 + 2 — inference orchestration + deployment artifacts built.** Phase 1: the self-balancing
GCS **shard-claim queue** (`inference/claim.py` atomic claim/heartbeat/done/reclaim, `scripts/shard_tiles.py`
spatial splitter, `scripts/run_inference_worker.py` one-per-GPU worker) + refactored the inference body into
`inference/runner.py` (`build_context`/`run_inference`, shared by CLI + worker) + the `scripts/inference_progress.py`
monitor/watcher (run + S2-export dashboards). Phase 2: built+verified+uploaded the **3 ensemble deployment
packages** (`gs://rts-mapping-v2-usw1/inference/2025q3_south/packages/seed{42,43,44}`), documented the
shard-scoped output-bucket layout, and **rebuilt + pushed `rts-infer:v1`** — self-contained (current code incl.
`inference/`, cv2 baked, MLflow 2.22.5; no runtime sed/mount). 309 tests green across the work. Prior:
Phase −1 training wrap-up (clean only-main, backups, inventory) + Phase-D deploy decision (ledger H/H.2/J).

<!-- NOW:BEGIN -->
### Now
**Inference phase — orchestration + artifacts done; gated on S2 + Phase-3 pre-flight.** Built & pushed:
shard-claim queue + worker + monitor (Phase 1), 3 ensemble deployment packages + self-contained `rts-infer:v1`
image (Phase 2). **Blocking long pole = the 2025_south Sentinel-2 export** (NDVI source): GEE-bound at
**3 concurrent task slots** → ~5 cells/hr, **~263/1799 done, ETA ~12 days** (diagnosed 2026-06-26; user chose
to wait, not cancel the competing 2024 export). **Phase 3 code done:** fleet scripts (`create_inference_fleet.sh` + `inference_fleet_startup.sh` +
`inference_watchdog.sh`) + live L4-quota check (32 limit / 1 phantom-used → 31 schedulable → default **3×
g2-standard-96**, not 4). **All remaining work is GATED:** (a) on the S2 export (~12 d) — `s2_index` +
coverage audit (Phase 0 tail), Banks Island RGB+NDVI parity, launch; (b) on explicit go + spend — live VM
pre-flight (1-VM startup smoke, multi-VM claim/kill drill, throughput benchmark → shard size + output dtype)
and Phase 4 launch. Shard the tile list at the benchmark-tuned size at launch (splitter ready; tile list
exists, not S2-gated). **Master = `a100-8x-train`, never stop/rename (A100 scarcity); shared PDG project —
only ever touch our `rts-`/`rts-infer-*` resources.**
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
