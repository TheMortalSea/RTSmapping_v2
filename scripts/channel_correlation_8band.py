"""8-band EXTRA channel correlation — on the *generated* v1.0 stack (Step 1 of the second-wave plan).

Unlike `scripts/channel_correlation.py` (which fetches 15 channels live from GEE), this reads the
already-generated 8-band EXTRA tiles from disk — i.e. the *exact* channels used in the Phase-4 ablation,
including the contrastive SE_PROTO and the 3 global-PCA components. Produces a clean 8x8 redundancy map to
overlay on the Step-3 synergy matrix.

Band order (data.md §9 / data/extra_channels.py): 0 NDVI, 1 NBR, 2-4 SE_PCA1-3, 5 SE_PROTO, 6 TCB, 7 TCW.

Three regimes, matching the prior study: all pixels (subsampled), positive (label==1), near-boundary (±10 px).

Usage (inside rts-train:v2, which has rasterio/scipy/seaborn):
  python scripts/channel_correlation_8band.py --n-tiles 150 --seed 42 \
      --extra-dir /outputs/v1.0/data_local/EXTRA --label-dir /outputs/v1.0/data_local/labels \
      --metadata /outputs/v1.0/data_local/metadata.csv --out-dir /outputs/v1.0/qc/correlation_8band
"""
import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

BAND_NAMES = ["NDVI", "NBR", "SE_PCA1", "SE_PCA2", "SE_PCA3", "SE_PROTO", "TCB", "TCW"]
N = len(BAND_NAMES)
ALL_PX_PER_TILE = 2000  # subsample for the all-pixel regime to bound memory
BOUNDARY_PX = 10


def _read(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    with rasterio.open(path) as ds:
        return ds.read()


def regime_pixels(extra: np.ndarray, label: np.ndarray, rng: np.random.Generator):
    """Return {regime: (P, 8)} pixel matrices for one tile. NaN rows dropped per regime later."""
    C, H, W = extra.shape
    flat = extra.reshape(C, -1).T  # (H*W, 8)
    lab = label.reshape(-1)
    out = {}
    # all: random subsample
    idx = rng.choice(flat.shape[0], size=min(ALL_PX_PER_TILE, flat.shape[0]), replace=False)
    out["all"] = flat[idx]
    # positive: label==1
    pos = lab == 1
    if pos.any():
        out["positive"] = flat[pos]
    # boundary: ±BOUNDARY_PX band around label==1
    m = (label == 1)
    if m.any():
        band = binary_dilation(m, iterations=BOUNDARY_PX) & ~binary_erosion(m, iterations=BOUNDARY_PX)
        bflat = band.reshape(-1)
        if bflat.any():
            out["boundary"] = flat[bflat]
    return out


def corr_matrix(px: np.ndarray, method: str) -> np.ndarray:
    """8x8 correlation, dropping non-finite rows. method in {spearman, pearson}."""
    good = np.isfinite(px).all(axis=1)
    px = px[good]
    if px.shape[0] < 50:
        return np.full((N, N), np.nan)
    if method == "spearman":
        r, _ = spearmanr(px)
        return np.atleast_2d(r)
    # pearson, guarding zero-variance columns
    std = px.std(axis=0)
    r = np.full((N, N), np.nan)
    ok = std > 1e-12
    if ok.sum() >= 2:
        sub = np.corrcoef(px[:, ok], rowvar=False)
        ii = np.where(ok)[0]
        for a, ia in enumerate(ii):
            for b, ib in enumerate(ii):
                r[ia, ib] = sub[a, b]
    return r


def heatmap(mat: np.ndarray, title: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(N)); ax.set_xticklabels(BAND_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N)); ax.set_yticklabels(BAND_NAMES)
    for a in range(N):
        for b in range(N):
            if np.isfinite(mat[a, b]):
                ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(mat[a, b]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="r", shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    log.info("wrote %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tiles", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra-dir", default="/outputs/v1.0/data_local/EXTRA")
    ap.add_argument("--label-dir", default="/outputs/v1.0/data_local/labels")
    ap.add_argument("--metadata", default="/outputs/v1.0/data_local/metadata.csv")
    ap.add_argument("--out-dir", default="/outputs/v1.0/qc/correlation_8band")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    extra_dir, label_dir = Path(args.extra_dir), Path(args.label_dir)

    # Positive tiles drive all 3 regimes (they carry RTS pixels + a boundary).
    meta = pd.read_csv(args.metadata)
    pos_ids = meta.loc[meta["TrainClass"] == "positive", "Tile_ID"].tolist()
    pos_ids = [t for t in pos_ids if (label_dir / f"{t}.tif").exists() and (extra_dir / f"{t}.tif").exists()]
    rng.shuffle(pos_ids)
    tiles = pos_ids[: args.n_tiles]
    log.info("using %d positive tiles (of %d available)", len(tiles), len(pos_ids))

    buckets: dict[str, list[np.ndarray]] = {"all": [], "positive": [], "boundary": []}
    for i, tid in enumerate(tiles):
        extra = _read(extra_dir / f"{tid}.tif")
        lab = _read(label_dir / f"{tid}.tif")
        if extra is None or lab is None:
            continue
        for regime, px in regime_pixels(extra, lab[0], rng).items():
            buckets[regime].append(px)
        if (i + 1) % 25 == 0:
            log.info("  %d/%d tiles", i + 1, len(tiles))

    mats = {}
    for regime, chunks in buckets.items():
        if not chunks:
            continue
        px = np.concatenate(chunks, axis=0)
        sp = corr_matrix(px, "spearman")
        mats[f"{regime}_spearman"] = sp
        heatmap(sp, f"Spearman — {regime} ({px.shape[0]:,} px)", out / f"heatmap_{regime}.png")
        if regime == "all":
            pe = corr_matrix(px, "pearson")
            mats["all_pearson"] = pe
            heatmap(pe, f"Pearson — all ({px.shape[0]:,} px)", out / "heatmap_all_pearson.png")

    # dendrogram + interpretations from the all-regime Spearman
    if "all_spearman" in mats:
        sp = mats["all_spearman"]
        dist = 1 - np.abs(sp)
        np.fill_diagonal(dist, 0.0)
        Z = linkage(squareform(dist, checks=False), method="average")
        plt.figure(figsize=(8, 4))
        dendrogram(Z, labels=BAND_NAMES, leaf_rotation=45)
        plt.ylabel("1 - |Spearman r|")
        plt.title("8-band EXTRA clustering (avg linkage)")
        plt.tight_layout()
        plt.savefig(out / "cluster_dendrogram.png", dpi=140)
        plt.close()

    # |r|>0.7 pairs across regimes
    rows = []
    for regime in ("all", "positive", "boundary"):
        key = f"{regime}_spearman"
        if key not in mats:
            continue
        m = mats[key]
        for a in range(N):
            for b in range(a + 1, N):
                if np.isfinite(m[a, b]) and abs(m[a, b]) > 0.7:
                    rows.append((BAND_NAMES[a], BAND_NAMES[b], round(float(m[a, b]), 3), regime))
    rows.sort(key=lambda r: -abs(r[2]))
    lines = ["# 8-band EXTRA correlation — |Spearman r| > 0.7", "", "| A | B | r | regime |", "|---|---|---|---|"]
    lines += [f"| {a} | {b} | {r} | {g} |" for a, b, r, g in rows]
    (out / "interpretations.md").write_text("\n".join(lines) + "\n")
    np.savez(out / "corr_matrices.npz", names=BAND_NAMES, **mats)
    log.info("DONE — %d high-corr pairs; artifacts in %s", len(rows), out)


if __name__ == "__main__":
    main()
