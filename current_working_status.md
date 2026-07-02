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
**Object-level scorecard instrument + bias/variance diagnosis (v3 pre-deploy, Phase 0).** Report-only scorecard
(`scripts/object_scorecard.py`; split/merge + matched-IoU geometry in `training.metrics._object_match_detail`;
per-region tile-cluster bootstrap CIs in `analyze_residual_errors.py`) — reproduces Finding K/J **exactly** on
the real val/test caches (self-check green). Ran on the 8× A100: in-sample train pass (`score_insample_train.py`,
3-seed ensemble over 302 train-positive tiles / 16 regions) + per-seed val (×3). **Result — perception-invisible
floor F_in=0.140 in-sample vs F_held=0.280 held-out** → the pre-registered rule lands **ambiguous band →
bake-off** (F_in is a *lower bound* — memorisation inflates fit — so the true bias floor is ≥14%; the ~14 pt
held-vs-in gap is a generalisation component). **Seed-noise floor** (Phase 0C, 3 existing seeds): obj-recall
std 0.028 / spread 0.052 — the margin any single-seed Phase-1 POC must beat. Splits/merges stay sub-dominant to
misses (recall remains the axis; 0E checkpoint doesn't reopen the lever). Artifacts + `decision_gate.json` in
`/mnt/outputs/v1.0/staging/object_scorecard_diagnostics/`; a 73-object invisible contact sheet + manifest
generated for the **D1 label audit (next v3 step)**. **D2 change-probe pending** — 2023 prior-year imagery not in
the local mount. 19 new tests green (torch+smp pinned to frozen deploy versions). Prior: pre-S2 residual-error
diagnostics (Finding K); inference Phases 1+2 (orchestration + 3 packages + `rts-infer:v1`).

<!-- NOW:BEGIN -->
### Now
**Multiscale POC (family M, branch `multiscale-poc`, started 2026-07-02 — user-approved plan):** prove
context-expanded 0.5x training fixes the zero-shot scale-transfer failure without hurting 1x. Phase A
done: `scripts/scale05_tile_creation.py` staged **21,934 tiles** (1,491 pos / 20,443 neg) at 9.55 m/px
to `gs://rts-mapping-v2/training/v1.0_scale05` (+ local copy); label rules = ignore auto-convert
(115/168) · unrefined-ARTS 255 guard · sub-pixel guard; QC green (contact sheet, norm drift −1..−7%,
splits resolve). Loader: `data.additional_roots` (train-only; val stays 1x-comparable). NDVI EXTRA
generating via GEE computePixels (interactive API — does NOT touch the S2 batch export queue). Next:
check_data → smoke → 3× A100 runs `multiscale_poc_seed{42,43,44}` → pre-registered gates (ledger
family M).
**(v3 pre-deploy side-track, done in parallel — does NOT gate the inference launch below):** object-scorecard
bias/variance diagnosis landed — F_in 14% (in-sample) / F_held 28% (held-out) → pre-registered rule =
**ambiguous / bake-off**; both a ≥14% representation-or-label bias floor and a ~14 pt generalisation gap are
present. Next v3 step = **D1 label audit** of the 73 in-sample invisibles (contact sheet ready in
`staging/object_scorecard_diagnostics/`); D2 change-probe still needs the 2023 prior-year imagery. Phase-1 POCs
(change / data arms) remain gated behind D1 + a decision to divert from deploy.

**Inference phase — orchestration + artifacts done; gated on S2 + Phase-3 pre-flight.** Built & pushed:
shard-claim queue + worker + monitor (Phase 1), 3 ensemble deployment packages + self-contained `rts-infer:v1`
image (Phase 2). **Blocking long pole = the 2025_south Sentinel-2 export** (NDVI source): GEE-bound, but
the 2024 competition has largely cleared → ~5 cells/hr realized, **1150/1799 done (64%, 2026-07-01), ETA
~5–6 days** (down from the ~12 d estimated 2026-06-26). South launcher finished submitting all tasks
(container exited — normal); GEE processes the rest server-side. **Phase 3 code done:** fleet scripts (`create_inference_fleet.sh` + `inference_fleet_startup.sh` +
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
