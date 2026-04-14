# Data Check

## Goal

Confirm the dataset in `gs://abrupt_thaw/RTS_MODEL_V2/DATA` matches `data/data.md`（ignore irrelevant files or folders）, and report dataset statistics.

## Deliverable

`scripts/check_data_bucket.py` — one script, flat, `argparse` + `logging`. Stream files via `/vsigs/` and `google-cloud-storage`; do not download the dataset.

Run:
```bash
python scripts/check_data_bucket.py --bucket gs://abrupt_thaw/RTS_MODEL_V2/DATA
```

Output: printed report + `docs/data_check_v{version}.log`. Exit 0 on pass, 1 on fail.

## Source of Truth

Read `data/data.md` first. If this file and `data.md` disagree, `data.md` wins — flag the conflict.

## Checks

Run all checks, aggregate results, do not fail fast.

1. **Layout** — bucket root has `PLANET-RGB/`, `EXTRA/`(optional), `labels/`, `metadata.csv`, `splits.yaml`, `version.json`(optional).
2. **Tile correspondence** — set of tile IDs is identical across `PLANET-RGB/`, `EXTRA/`, `labels/`, and `metadata.csv`. Report mismatches.
3. **metadata.csv schema** — columns match `data.md` §3.3 exactly; `Tile_id` unique; `TrainClass ∈ {Positive, Negative}`; `UIDs` empty iff `TrainClass=Negative`; `RegionName` non-empty.
4. **Splits** — every region in `splits.yaml` exists in `metadata.csv`; no region in more than one split; flag any metadata region unassigned to a split.
5. **Rasters (sample all positives + 200 random negatives, seed 42)** — for each sampled `tile_id`:
   - RGB: `(3, 512, 512)` uint8, values in `[0, 255]`, no NaN.
   - EXTRA: `(4, 512, 512)`, all finite.
   - Label: `(512, 512)` uint8, values ⊂ `{0, 1, 255}`.
   - CRS = `EPSG:3857`; bounds and transform identical across the three files of each tile.
6. **Label semantics (same sample)** — `Positive` tiles have ≥1 pixel equal to `1`; `Negative` tiles have zero pixels equal to `1`.

## Statistics to Report

Compute and print (also save alongside the log):

**Tile counts**
- Total tiles, positive tiles, negative tiles, positive:negative ratio.
- Tiles per split (train/val/test), with positive/negative breakdown per split.
- Tiles per region, with positive/negative breakdown.

**Polygon / RTS counts**
- Total unique RTS UIDs across all Positive tiles (parse `UIDs` column).

**Label pixel statistics (on sampled rasters)**
- Number of Positive tiles with 1–10%, 10–50%, >50% RTS pixel coverage.

**Data version**
- Contents of `version.json`.