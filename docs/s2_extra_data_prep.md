# Sentinel-2 Download & EXTRA-Channel Data Preparation — Design Doc

**Status:** Proposal for review (Heidi & Robb)
**Author:** RTS mapping team
**Scope:** Data download + preprocessing only. No model building, no inference runs.
**Decision needed:** Sign-off on the approach + the five open decisions in §6.

---

## 1. Why this doc

We are about to acquire and process the Sentinel-2 (S2) imagery that two models depend on, and the
choices made now (extent, CRS, bands, bucket, grid) are expensive to redo once terabytes are on disk.
This doc proposes *how* we download and preprocess that data so reviewers can catch problems before we
spend storage budget. **We run the whole pipeline ourselves — there is no external handoff.**

Two models drive the requirements:

1. **Planet model** — the existing RTS segmentation model (PlanetScope RGB + auxiliary "EXTRA"
   channels), trained on 2024, deployed on **2025** PlanetScope basemaps over the pan-Arctic Planet
   domain (~60–74°N). It needs **2025 inference EXTRA channels** generated identically to training.
2. **Pure-S2 model** *(new)* — an RTS model that runs on Sentinel-2 directly, to cover
   **74–84°N (ARTS North)** where **no PlanetScope basemap exists**. Trained on **2024** S2 against the
   current ARTS labels; deployed on **2025** S2.

---

## 2. The data matrix (what we need)

Two regions × two years, with different jobs in each cell:

```
                 │  2024  (TRAINING-side)                │  2025  (INFERENCE-side)
─────────────────┼───────────────────────────────────────┼───────────────────────────────────────
 ARTS NORTH      │  ✅ HAVE  (in EPSG:3413 — re-export)    │  ⬇ DOWNLOAD
 74–84°N         │  → pure-S2 model TRAINING              │  → pure-S2 model INFERENCE
 (no Planet)     │                                       │
─────────────────┼───────────────────────────────────────┼───────────────────────────────────────
 ARTS SOUTH      │  ⬇ DOWNLOAD (South)                    │  ⬇ DOWNLOAD (full Planet coverage)
 (Planet domain) │  → EXTRA-for-training source          │  → Planet-model EXTRA INFERENCE
                 │  → pure-S2 model TRAINING             │  → pure-S2 model INFERENCE
```

Same thing as a table:

| Region | Year | Status | Purpose(s) |
|---|---|---|---|
| ARTS North (74–84°N) | 2024 | ✅ in GCS (EPSG:3413, 398 tiles) → **re-export to 3857** | pure-S2 model **training** |
| ARTS North (74–84°N) | 2025 | ⬇ download | pure-S2 model **inference** |
| ARTS South (Planet domain) | 2024 | ⬇ download | EXTRA-for-training source; pure-S2 model **training** |
| ARTS South (Planet domain) | 2025 | ⬇ download (full coverage) | Planet-model **EXTRA inference**; pure-S2 model **inference** |

**Rule of thumb:** *2024 = training-side, 2025 = inference-side.*

### Data flow

```
                         ┌──────────────────────────────────────────────┐
   GEE: S2_SR_HARMONIZED │  Bulk S2 median composites (per 1°×3° tile)   │
   + QA60 mask, summer   │  → GCS, COG, EPSG:3857   (§3)                 │
   median composite      └───────────────┬──────────────────────────────┘
                                          │
                 ┌────────────────────────┴───────────────────────────┐
                 ▼                                                      ▼
   ┌──────────────────────────────┐                  ┌──────────────────────────────────┐
   │ Pure-S2 model tiles (§5)     │                  │ Planet EXTRA channels (§4)        │
   │ 512×512, EPSG:3857           │                  │ per Planet-RGB footprint, EPSG:3857│
   │ 2024+labels → train          │                  │ via generate_extra_tiles.py        │
   │ 2025 → inference             │                  │ (reuses GEE directly; see §4)      │
   └──────────────────────────────┘                  └──────────────────────────────────┘
```

> Note: the EXTRA pipeline (§4) queries GEE per Planet-RGB footprint directly; it does **not** consume
> the bulk S2 composites of §3. The bulk composites of §3 are for the **pure-S2 model** (and as an
> on-disk S2 reference). They share the same GEE source/recipe, so results are consistent.

---

## 3. Track 1 — Bulk Sentinel-2 composite export

**Approach.** Port our two existing Colab/GEE notebooks (grid generation + gridded S2 export) into
resumable, non-Colab scripts that run on the VM (§7), changing the export CRS to **EPSG:3857**.

- **Grid:** clean **1°×3°** latitude×longitude grid, land-filtered against LSIB
  (`USDOS/LSIB_SIMPLE/2017`), the same GENERATED approach already used for ARTS North. We **drop** the
  earlier "aggregate the Planet fine grid" path — it produced oversized cells (1.3°×3.9°) and its
  edge-merge was an unimplemented stub.
- **Compositing recipe** (unchanged from the working notebook): `COPERNICUS/S2_SR_HARMONIZED`, QA60
  cloud+cirrus mask, summer window **DOY 180–273**, `CLOUDY_PIXEL_PERCENTAGE < 5`, **median** composite,
  exported via `Export.image.toCloudStorage` as deflate-compressed Cloud-Optimized GeoTIFF.
- **CRS:** **EPSG:3857 everywhere** (project standard). We accept that Web Mercator is distorted at
  74–84°N; this keeps one pipeline and matches the Planet training/inference tiles. *(Alternative —
  EPSG:3413 for the North — was considered and rejected for pipeline simplicity; see §6.3.)*
- **Bands:** **TBD — placeholder pending the channel-selection experiments.** We will lock the band
  list after those results. Expected to include at least RGB + NIR + NDVI; SWIR (B11/B12) added only if
  NBR / Tasseled-Cap make the final EXTRA list.
- **Resumability:** skip tiles already present in the GCS folder and tasks already active in the GEE
  queue; back off and retry on "Too many tasks".

**Runs:**

| Run | Region/Year | Notes |
|---|---|---|
| `s2_2024_south` | ARTS South 2024 | training-side; extent per §6.1 |
| `s2_2025_south` | ARTS South 2025 | inference-side; **full Planet coverage** |
| `s2_2025_north` | ARTS North 2025 | inference-side |
| `s2_2024_north_reexport` | ARTS North 2024 | re-export existing 3413 tiles to EPSG:3857 (§6.3) |

---

## 4. Track 2 — Planet-model inference EXTRA channels

**Reuse, don't rebuild.** `scripts/generate_extra_tiles.py` + `data/extra_channels.py` (already merged)
produce the per-tile EXTRA stack in EPSG:3857 from each PLANET-RGB footprint, querying GEE
(Sentinel-2 + Google Satellite Embedding). It is resumable, multi-threaded, and parameterized by
`--year / --groups / --rgb-dir / --out-dir` — **it was written to be exactly this inference path**
("2024 training and 2025 inference tiles produced identically").

For 2025 inference we drive it with `--year 2025` over the 2025 Planet-RGB tiles:

```
python scripts/generate_extra_tiles.py --groups <s2|all> --year 2025 \
   --metadata <2025 inference tile list> --rgb-dir <2025 PLANET-RGB tiles> \
   --out-dir  <.../EXTRA_2025> [--se-artifacts se_artifacts.npz] --workers N
```

- **Gated on the final EXTRA channel list** (channel-selection experiments are still running):
  - NDVI / S2-derived only → `--groups s2`, **no AlphaEarth needed**.
  - If Satellite-Embedding (SE) channels make the cut → also need `se_artifacts.npz` +
    AlphaEarth 2025 availability → `--groups all`.
- **Prerequisite (ours):** the **2025 Planet-RGB inference tiles** define the footprints, so they must
  be tiled first. We own this ingest — see §6.5.
- **Scale:** ~3.4M tiles is the cost driver. The runner is thread-pooled and resumable; we shard the
  tile list across the VM's cores (and, if needed, across multiple VMs).

---

## 5. Track 3 — Pure-S2 model data prep (prep only)

*(Building/training the model itself is out of scope; this is just its data.)*

- **Training tiles:** co-register **S2-2024** (North + South) with the **current ARTS labels** into
  512×512 EPSG:3857 image/label tiles, reusing the tiling logic in
  `scripts/positive_tile_creation.py` / `scripts/negative_tile_creation.py` with the S2 composite
  swapped in for the Planet source. Label convention unchanged: 0 = background, 1 = RTS, 255 = ignore.
- **Inference tiles:** tile **S2-2025** (North + South) to 512×512.
- **Normalization:** per-dataset stats via `scripts/compute_normalization_stats.py` for the S2 bands.

---

## 6. Open decisions (please weigh in)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| 6.1 | **2024 South extent** | label-bearing Planet South only *vs* full Planet South | Label regions only (cheaper; 2024 is training-side) |
| 6.2 | **Final S2 bands** | — | **Placeholder — lock after channel-selection experiments** |
| 6.3 | **North-2024 CRS fix** | re-export in 3857 *vs* reproject existing 3413 tiles | Re-export (avoids resampling artifacts; cheap via GEE) |
| 6.4 | **Bucket** | keep `gs://pdg-storage-default/sentinel2/…` *vs* co-locate in PDG `gs://rts-mapping-v2` (VM region) | Co-locate in `rts-mapping-v2` to cut cross-project/region egress |
| 6.5 | **2025 Planet-RGB ingest** (Track-2 prerequisite, ours) | fold into this effort now *vs* sequence as a separate follow-up | Sequence as a near-term follow-up, ahead of Track 2 |

---

## 7. Compute, storage & cost

- **VM:** one dedicated **CPU** VM (the GEE export runs server-side; EXTRA generation and tiling are
  embarrassingly parallel I/O + light CPU). High-vCPU, **Spot/preemptible**, no GPU; stopped when idle.
  Provision per [computing/vm_instruction.md](../computing/vm_instruction.md).
- **Storage is the watch-item.** The $70k PDG credit is **compute-only** — storage and network egress
  are billed separately. A 1°×3° S2 composite at 10 m is ≈ 11k×11k px × several bands ≈ **>1 GB/tile**;
  full 40–84°N across two years is **multi-TB**. Mitigations: deflate-compressed COGs, trimming the
  2024 extent to label regions (§6.1), and co-locating the bucket with the VM (§6.4). Infra facts and
  budget live in [computing/infrastructure.md](../computing/infrastructure.md).

---

## 8. Out of scope

Building/training either model; running Planet or S2 inference; the 2025 Planet-RGB basemap ingest
itself (a Track-2 prerequisite — ours, sequenced separately per §6.5); rebuilding AlphaEarth artifacts
unless SE channels make the final EXTRA list. All of these are ours to do later — this doc covers only
the S2 download + EXTRA/S2-model data preparation design.
