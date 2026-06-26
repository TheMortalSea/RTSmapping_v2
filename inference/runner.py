"""Shared inference setup + per-tile loop (inference.md §8).

Both the single-shot CLI (`scripts/inference.py`) and the queue worker
(`scripts/run_inference_worker.py`) use this module so the model setup, NDVI
windowing, ensemble fusion, NoData handling, and COG writing are defined exactly
once (CLAUDE Rule 3). The CLI runs `run_inference` once over a whole tile list;
the worker calls it per claimed shard, reusing one `InferenceContext`.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from inference.predictor import (
    assert_runtime_matches_package, load_deployment_package, predict_probs,
    predict_probs_ensemble,
)
from inference.quad_index import load_quad_index
from inference.s2_index import load_s2_index
from inference.tiles import InferenceTileDataset
from inference.writer import NODATA_PROB, Manifest, write_probability_tile

logger = logging.getLogger(__name__)


def _collate(items: list[dict]) -> dict:
    """Stack a batch, keeping per-tile metadata as lists."""
    return {
        "tile_id": [it["tile_id"] for it in items],
        "image": torch.from_numpy(np.stack([it["image"] for it in items])),
        "nodata_mask": np.stack([it["nodata_mask"] for it in items]),
        "all_nodata": [it["all_nodata"] for it in items],
        "bounds": [tuple(it["bounds"]) for it in items],
    }


def weights_sha256(package: str) -> str:
    """SHA256 of a package's weights.pth (local or gs://) for the manifest."""
    path = f"{package.rstrip('/')}/weights.pth"
    h = hashlib.sha256()
    if path.startswith("gs://"):
        import gcsfs
        f = gcsfs.GCSFileSystem(token="google_default").open(path[5:], "rb")
    else:
        f = open(path, "rb")
    with f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class InferenceContext:
    """Heavy, run-wide setup loaded once and reused for every shard/tile list."""

    models: list
    pkg: dict
    dep_cfg: dict
    run_cfg: dict
    quad_index: pd.DataFrame
    s2_index: Optional[pd.DataFrame]
    extra_bands: list
    ensemble: bool
    package_paths: list[str]


def build_context(config: dict, packages: list[str], quad_index_path: str,
                  s2_index_path: Optional[str], device: torch.device) -> InferenceContext:
    """Load packages + indices and validate the ensemble/calibration contract.

    Mirrors the §8.2 init: load each deployment package, assert ensemble members
    share calibration + channel layout, assert runtime matches the package (§14),
    and load the quad index (+ S2 index when the package declares EXTRA=NDVI).
    """
    pkgs = [load_deployment_package(pp, device) for pp in packages]
    pkg = pkgs[0]                       # reference for stats / model_cfg / dep_cfg
    dep_cfg = pkg["dep_cfg"]
    ensemble = len(pkgs) > 1
    if ensemble:
        # Members must share the calibration + channel layout (fusion is on the
        # final calibrated prob); otherwise the fused threshold is meaningless.
        for other in pkgs[1:]:
            for k in ("temperature", "threshold", "tta", "precision"):
                if other["dep_cfg"].get(k) != dep_cfg.get(k):
                    raise ValueError(f"ensemble member {k} mismatch: "
                                     f"{other['dep_cfg'].get(k)} != {dep_cfg.get(k)}")
            if other["n_channels"] != pkg["n_channels"]:
                raise ValueError("ensemble member channel count mismatch")
        logger.info("ENSEMBLE inference: %d members, T=%.4f thr=%.3f tta=%s",
                    len(pkgs), dep_cfg["temperature"], dep_cfg["threshold"],
                    dep_cfg.get("tta", "none"))
    assert_runtime_matches_package(config, dep_cfg)

    # EXTRA=NDVI is windowed from the bulk S2 composites on the fly (inference.md §5).
    extra_bands = (pkg["model_cfg"].get("channels") or {}).get("extra") or []
    s2_index = None
    if extra_bands:
        if not s2_index_path:
            raise ValueError("package declares EXTRA channels but no s2_index was "
                             "provided (needed to window NDVI from the S2 composites)")
        s2_index = load_s2_index(s2_index_path)

    quad_index = load_quad_index(quad_index_path)
    return InferenceContext(
        models=[p["model"] for p in pkgs], pkg=pkg, dep_cfg=dep_cfg, run_cfg=config,
        quad_index=quad_index, s2_index=s2_index, extra_bands=extra_bands,
        ensemble=ensemble, package_paths=[str(pp) for pp in packages])


def run_metadata(ctx: InferenceContext, device: torch.device) -> dict:
    """Assemble the §9.4 inference_log.json run metadata for this context."""
    dep_cfg, run_cfg = ctx.dep_cfg, ctx.run_cfg
    return {
        "model_version": "+".join(Path(pp.rstrip("/")).name for pp in ctx.package_paths),
        "deployment_package_path": ctx.package_paths,
        "model_checkpoint_sha": [weights_sha256(pp) for pp in ctx.package_paths],
        "ensemble_members": len(ctx.models),
        "inference_date": datetime.now(timezone.utc).isoformat(),
        "scales_used": dep_cfg.get("scales", [1.0]),
        "tta_config": dep_cfg.get("tta", "none"),
        "precision": dep_cfg.get("precision"),
        "torch_compile": bool(dep_cfg.get("torch_compile", False)),
        "threshold": dep_cfg["threshold"],
        "temperature": dep_cfg["temperature"],
        "stride_px": run_cfg["inference"]["stride_px"],
        "overlap_aggregation": "gaussian_weighted_mean",
        "fusion_sigma_px": run_cfg["inference"]["fusion_sigma_px"],
        "gpu_type": (torch.cuda.get_device_name(device)
                     if device.type == "cuda" else "cpu"),
    }


def run_inference(ctx: InferenceContext, tiles: pd.DataFrame, output: str,
                  manifest: Manifest, device: torch.device, num_workers: int = 8,
                  scale: float = 1.0,
                  progress_cb: Optional[Callable[[int, int], None]] = None) -> dict:
    """Run §8.2 inference over ``tiles``, writing probability COGs under ``output``.

    Resumable via ``manifest`` (skips tiles already recorded). ``progress_cb`` is
    invoked as ``(n_done, n_total)`` at each progress-log point — the queue worker
    uses it to heartbeat its shard claim. Returns the manifest counts.
    """
    out = output.rstrip("/")
    todo = tiles[~tiles["tile_id"].astype(str).isin(manifest.completed)]
    logger.info("%d tiles total, %d already done, %d to process",
                len(tiles), len(tiles) - len(todo), len(todo))
    if todo.empty:
        manifest.save()
        return manifest.counts()

    pkg, dep_cfg = ctx.pkg, ctx.dep_cfg
    dataset = InferenceTileDataset(todo, ctx.quad_index, pkg["stats"], scale=scale,
                                   s2_index=ctx.s2_index, extra_bands=ctx.extra_bands)
    loader = DataLoader(dataset, batch_size=ctx.run_cfg["inference"]["batch_size"],
                        num_workers=num_workers, collate_fn=_collate)

    t0, n_done = time.time(), 0
    for batch in loader:
        keep = [i for i, all_nd in enumerate(batch["all_nodata"]) if not all_nd]
        for i, all_nd in enumerate(batch["all_nodata"]):
            if all_nd:
                manifest.mark(batch["tile_id"][i], "all_nodata")
        if keep:
            images = batch["image"][keep].to(device)
            if ctx.ensemble:
                probs = predict_probs_ensemble(ctx.models, images,
                                               temperature=dep_cfg["temperature"],
                                               tta=dep_cfg.get("tta", "none"),
                                               precision=dep_cfg.get("precision", "fp32"))
            else:
                probs = predict_probs(pkg["model"], images,
                                      temperature=dep_cfg["temperature"],
                                      tta=dep_cfg.get("tta", "none"),
                                      precision=dep_cfg.get("precision", "fp32"))
            probs = probs.clamp_(0.0, 1.0).cpu().numpy()  # §10.1 range guard
            for j, i in enumerate(keep):
                prob = probs[j]
                prob[batch["nodata_mask"][i]] = NODATA_PROB  # §5.3 output mask
                tile_id = batch["tile_id"][i]
                write_probability_tile(f"{out}/{tile_id}.tif", prob,
                                       batch["bounds"][i])
                manifest.mark(tile_id, "done")
        n_done += len(batch["tile_id"])
        rate = n_done / (time.time() - t0)
        if n_done % 512 < len(batch["tile_id"]):
            logger.info("%d/%d tiles (%.1f tiles/s, ETA %.1f h)",
                        n_done, len(todo), rate, (len(todo) - n_done) / rate / 3600)
            if progress_cb is not None:
                progress_cb(n_done, len(todo))

    manifest.save()
    logger.info("Done: %s", manifest.counts())
    return manifest.counts()
