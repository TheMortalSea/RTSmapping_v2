# Artifact & Bucket Inventory

**Single source of truth for "what artifact lives where."** Every durable artifact produced since project
start → its bucket/path, region, owning project, and whether it's source-of-truth or derived/backup.
Companion to `infrastructure.md §4` (bucket facts) — this doc is the *artifact → location* map.

> Surveyed 2026-06-26. When you produce a new durable artifact or move one, update this table.

---

## 1. Buckets at a glance

| Bucket | Project | Region | Role |
|---|---|---|---|
| `gs://abrupt_thaw/` | abruptthawmapping (non-PDG) | US multi-region | **Original/legacy data** + the v2-alpha training source (`RTS_MODEL_V2/DATA/`). Reading from PDG VMs crosses projects → egress. |
| `gs://rts-mapping-v2/` | PDG | US multi-region | **Compute-adjacent SSoT** — training data (`training/v1.0/`), run + MLflow mirrors (`RTS_MODEL_V2/`), and the planned `backups/`. |
| `gs://rts-mapping-v2-usw1/` | PDG | **us-west1** | **Sentinel-2 composites** (`S2_RGB/{2024_train,2025_south}`) — NDVI source + S2 model. **Inference I/O** prefix (to create): `inference/2025q3_south/`. |
| `gs://pdg-planet-data/` | PDG | **us-west1** | **2025 Planet basemap quads** (`global_quarterly/`) — pan-arctic inference input. **Shared PDG data — read-only, not ours.** |

---

## 2. Artifact → location

| Artifact | Where (SSoT) | Region/proj | Backup / mirror | Notes |
|---|---|---|---|---|
| **Training imagery + labels (v1.0)** | `gs://rts-mapping-v2/training/v1.0/{PLANET-RGB,labels,metadata.csv,splits.yaml,normalization_stats*.json}` | PDG / US | local `/mnt/outputs/v1.0/data_local/` | The working training set. `abrupt_thaw/RTS_MODEL_V2/DATA/` is the upstream/legacy source. |
| **Legacy arrays** | `gs://abrupt_thaw/{maxar_rgb,rts_labels}*.npy`, `CAMS/`, hashed dirs | abruptthaw / US | — | Pre-v2 / exploratory; not on the v2 path. |
| **Training runs (checkpoints, configs, logs, figures)** | local `/mnt/outputs/v1.0/runs/<run>/` | A100 master (local disk) | `gs://rts-mapping-v2/RTS_MODEL_V2/runs/` | **110 local vs 105 in GCS — 5 recent Phase-D runs unmirrored** (sync in Phase −1.3). |
| **MLflow tracking** | local `/mnt/outputs/v1.0/mlflow/` + `/mnt/outputs/mlflow/` | A100 master | `gs://rts-mapping-v2/RTS_MODEL_V2/mlflow/` | UI served from the master. |
| **Calibration report** (Phase D) | local `/mnt/outputs/v1.0/calibration/effb5_trivialaug/` | A100 master | **none yet — local-only** | Back up in −1.3. |
| **Test-Realistic metrics** (shipped #) | local `/mnt/outputs/v1.0/test_realistic/effb5_ensemble_metrics.json` | A100 master | **none yet — local-only** | The one-shot v2 number (ledger J). Back up in −1.3. |
| **Object operating-point report + val_probs** | local `/mnt/outputs/v1.0/object_operating_point/effb5_ensemble/` | A100 master | **none yet — local-only** | `val_probs.npz` ~1.1 GB. Back up in −1.3. |
| **Deployment packages** (3 seeds) | **not built yet** → Phase 2 → `gs://rts-mapping-v2-usw1/inference/2025q3_south/packages/` (or `rts-mapping-v2/.../models/`) | PDG / us-west1 | — | Built in Phase 2 from the run dirs. |
| **Inference tile list + quad index** | local `/mnt/outputs/inference/{tiles_2025q3_domain_full.csv,quad_index_2025q3.csv}` | A100 master | back up in −1.3 | 41.57M tiles / 309,101 quads. |
| **S2 index** (NDVI windowing) | **not built yet** → Phase 0 → `gs://rts-mapping-v2-usw1/inference/2025q3_south/s2_index.csv` | PDG / us-west1 | — | From `scripts/build_s2_index.py`. |
| **Inference output** (prob COGs, claims, manifests) | **run output** → `gs://rts-mapping-v2-usw1/inference/2025q3_south/{probs,shards,claims,done,logs}/` | PDG / us-west1 | (is the durable product) | Float32 COGs; ~12 TB est. Hierarchical shard-scoped prefixes (pre-mortem #1). |
| **Docker image** | `us-west1-docker.pkg.dev/pdg-project-406720/pdg-artifact-registry/rts-train:v2` (training) → `rts-infer:v1` (Phase 2) | PDG Artifact Registry / us-west1 | — | Built locally + pushed via ADC (Cloud Build blocked, §8). |
| **Docs / ledger / report** | repo (`docs/`, `current_working_status.md`) + GitHub `whrc/RTSmapping_v2` | — | git | `report.html` is gitignored (regenerated). |

---

## 3. Local `/mnt/outputs` (A100 master — local disk, NOT durable)

Source-of-truth (back up): `v1.0/runs`, `v1.0/mlflow`, `v1.0/calibration`, `v1.0/test_realistic`,
`v1.0/object_operating_point`, `v1.0/staging` (normalization stats), `inference/` (tile lists, quad index).
Derived/disposable: `hf_cache/`, `bench/`, `s2_qc/`, `worktrees/`, `_archive/`, scratch (`_du.txt`,
`_tmpcheck.txt`, `_paper.txt`, `upload(irrelevant)/`), S2 export logs. See `/mnt/outputs/README.md`
(rewritten in Phase −1.3) for the cleaned layout.
