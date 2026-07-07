"""Load the Banks Island QC package into ArcGIS Pro (post-inference.md's
review step): the merged probability/mask rasters, the RTS polygon layer, and
the RGB "underlying tile" context chips built by scripts/build_rgb_chips.py.

**Windows-only, run inside ArcGIS Pro's own Python environment** (arcpy is not
importable anywhere else — there is no Linux/CI path for this script). Two
ways to run it:

  1. Open ArcGIS Pro, open (or start a new blank) project, then in the
     Python window (a plain Python console — no IPython, so no `%run` magic):
        ```
        import sys
        sys.path.insert(0, r"E:\path\to\this\script's\folder")
        import build_arcgis_project as bap
        sys.argv = ["build_arcgis_project.py",
                    "--products-dir", r"D:\rts_qc\banks",
                    "--products-uri", "gs://rts-mapping-v2-usw1/inference/banks/products/"]
        bap.main()
        ```
     — this adds layers to the *currently open* project/map.
  2. From the "Python Command Prompt" ArcGIS Pro installs (Start Menu ->
     ArcGIS -> Python Command Prompt; activates arcgispro-py3), pointing at
     an existing .aprx (no open Pro session needed, so `--project` is
     required — "CURRENT" only resolves inside a live Pro session):
     `python build_arcgis_project.py --products-dir D:\rts_qc\banks --project D:\rts_qc\banks.aprx`

Either way, pass `--products-uri` to have it pull the whole products/ prefix
(probability.tif, mask.tif, banks_rts.gpkg, region_log.json, rgb_chips/,
rgb_chips.vrt) down to `--products-dir` first via `gcloud storage rsync`
(requires the Google Cloud SDK on the Windows machine — the same one you
already use to reach this project's buckets).

Usage:
    python build_arcgis_project.py --products-dir D:\\rts_qc\\banks ^
        --products-uri gs://rts-mapping-v2-usw1/inference/banks/products/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import arcpy
except ImportError:
    sys.exit(
        "arcpy is not importable. This script must be run inside ArcGIS "
        "Pro's own Python environment (arcgispro-py3), not a plain Windows "
        "Python install — see the module docstring for the two supported "
        "invocation modes."
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def download_products(products_uri: str, products_dir: Path) -> None:
    """One-shot pull of the whole products/ prefix via the GCS CLI.

    Resolves the executable via ``shutil.which`` rather than passing the bare
    "gcloud" to subprocess: on Windows gcloud is installed as ``gcloud.cmd``,
    and unlike a real shell, Python's subprocess doesn't consult PATHEXT for a
    bare name — it needs the resolved path (WinError 2 otherwise).
    """
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise RuntimeError("gcloud not found on PATH — install the Google Cloud SDK")
    products_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [gcloud, "storage", "rsync", "-r", products_uri, str(products_dir)],
        check=True,
    )
    logger.info("Synced %s -> %s", products_uri, products_dir)


def find_rts_layer_source(gpkg_path: Path) -> str:
    """Discover the RTS feature class inside the gpkg (name varies by region)."""
    arcpy.env.workspace = str(gpkg_path)
    fcs = arcpy.ListFeatureClasses()
    if not fcs:
        raise RuntimeError(f"No feature classes found in {gpkg_path}")
    return f"{gpkg_path}\\{fcs[0]}"


def add_layers(m, products_dir: Path):
    """Add the 4 QC layers bottom -> top; returns the RTS polygon layer."""
    rgb_lyr = m.addDataFromPath(str(products_dir / "rgb_chips.vrt"))
    prob_lyr = m.addDataFromPath(str(products_dir / "probability.tif"))
    mask_lyr = m.addDataFromPath(str(products_dir / "mask.tif"))
    rts_source = find_rts_layer_source(products_dir / "banks_rts.gpkg")
    rts_lyr = m.addDataFromPath(rts_source)

    try:
        prob_sym = prob_lyr.symbology
        prob_sym.colorizer.stretchType = "StandardDeviation"
        ramps = m.listColorRamps("Yellow-Orange-Red (Continuous)")
        if ramps:
            prob_sym.colorizer.colorRamp = ramps[0]
        prob_lyr.symbology = prob_sym
        prob_lyr.transparency = 40
    except Exception:
        logger.warning("Could not set probability.tif symbology (Pro-version "
                        "API drift) — layer added with default symbology.",
                        exc_info=True)

    mask_lyr.visible = False  # redundant with the vector layer; kept for QC

    try:
        rts_sym = rts_lyr.symbology
        rts_sym.renderer.symbol.applySymbolFromGallery("Black Outline (1pt)")
        rts_sym.renderer.symbol.color = {"RGB": [255, 0, 0, 100]}
        rts_sym.renderer.symbol.outlineColor = {"RGB": [255, 0, 0, 100]}
        rts_sym.renderer.symbol.size = 0
        rts_lyr.symbology = rts_sym
    except Exception:
        logger.warning("Could not set banks_rts.gpkg symbology (Pro-version "
                        "API drift) — layer added with default symbology.",
                        exc_info=True)

    logger.info("Added layers: %s, %s, %s, %s", rgb_lyr.name, prob_lyr.name,
                mask_lyr.name, rts_lyr.name)
    return rts_lyr


def zoom_to_layer(aprx, layer) -> None:
    """Best-effort zoom to the RTS layer's extent (needs an active map view)."""
    try:
        view = aprx.activeView
        extent = view.getLayerExtent(layer, False, True)
        view.camera.setExtent(extent)
    except Exception:
        logger.warning("Could not zoom to the RTS layer extent — no active "
                        "map view, or a Pro-version API mismatch; navigate "
                        "manually.", exc_info=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--products-dir", required=True, type=Path,
                   help="local dir to hold/read probability.tif, mask.tif, "
                        "banks_rts.gpkg, rgb_chips/, rgb_chips.vrt")
    p.add_argument("--products-uri", default=None,
                   help="gs:// products prefix to sync down first; omit to "
                        "use --products-dir as already-downloaded")
    p.add_argument("--project", default=None,
                   help="path to an existing .aprx to open; omit to use the "
                        "currently-open ArcGIS Pro project (CURRENT)")
    p.add_argument("--map-name", default=None,
                   help="map to add layers to; omit to use the first map")
    args = p.parse_args()

    if args.products_uri:
        download_products(args.products_uri, args.products_dir)

    aprx = arcpy.mp.ArcGISProject(args.project or "CURRENT")
    maps = aprx.listMaps(args.map_name) if args.map_name else aprx.listMaps()
    m = maps[0] if maps else aprx.createMap("RTS QC")
    # A newly-created map (or one that just isn't the active view) exists in
    # the project but isn't necessarily visible — openView() activates it so
    # the layers we're about to add actually show up on screen.
    m.openView()

    rts_lyr = add_layers(m, args.products_dir)
    zoom_to_layer(aprx, rts_lyr)
    aprx.save()
    logger.info("Saved %s", aprx.filePath)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
