"""Generate the project's single living HTML report from MLflow.

The report (`docs/report.html`) is the project dashboard — all findings, past/current/future,
accumulate here. Rules (frequency / contents / format) are in `docs/report.md`; the experiment
program SSoT is `training/experiments.md`. Sections (each auto-populates from MLflow by run_name
prefix, or shows a pending/blocked/gated badge):
  1. Overview & status      5. Phase 3 — loss family → boundary
  2. Phase 0 — baseline      6. Phase 4 — EXTRA channels
  3. Phase 1 — temporal      7. Phase 5 — architecture (gated)
  4. Phase 2 — data scaling  8. Findings & insights   9. Open questions / future

Usage:
    python scripts/build_report.py \\
        --config configs/baseline.yaml \\
        --output docs/report.html

Requirements: mlflow, pandas, matplotlib (all in requirements.txt).
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config import load_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _connect_mlflow(tracking_uri: str):
    import mlflow
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                          os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def _search_runs(mlflow, experiment_name: str, prefix: str) -> pd.DataFrame:
    """Return all runs whose run_name starts with `prefix`."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"tags.mlflow.runName LIKE '{prefix}%'",
            order_by=["tags.mlflow.runName ASC"],
        )
    except Exception as exc:
        logger.warning("MLflow search failed (%s); returning empty frame.", exc)
        runs = pd.DataFrame()
    return runs


def _get_metric_history(mlflow, run_id: str, metric: str) -> list[tuple[int, float]]:
    """Return [(step, value), ...] for a metric in a run."""
    try:
        client = mlflow.MlflowClient()
        history = client.get_metric_history(run_id, metric)
        return [(h.step, h.value) for h in history]
    except Exception:
        return []


def _download_artifact_text(mlflow, run_id: str, artifact_name: str) -> str | None:
    """Download a small text artifact from MLflow and return its content."""
    try:
        client = mlflow.MlflowClient()
        local = client.download_artifacts(run_id, artifact_name)
        return Path(local).read_text()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _img_file_b64(path: str) -> str | None:
    """Read a PNG file → base64, or None if missing. Tries /outputs and /mnt/outputs."""
    for p in (Path(path), Path(path.replace("/outputs/", "/mnt/outputs/"))):
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return None


# Qualitative-artifact paths (container /outputs; helper falls back to /mnt/outputs).
_VAL_OVERLAY = "/outputs/v1.0/inference/v1.0_baseline_validation/validation_overlay.png"
_VAL_OVERLAY_2024 = "/outputs/v1.0/inference/v1.0_baseline_validation/validation_overlay_2024labels.png"
_EXTRA_VIS = "/outputs/v1.0/qc/extra_vis"


def _plot_metric_curves(
    histories: dict[str, list[tuple[int, float]]],
    title: str,
    xlabel: str = "Epoch",
    ylabel: str = "",
    colors: list[str] | None = None,
) -> str:
    """Plot multiple (step, value) histories; return base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = colors or ["#2563EB", "#DC2626", "#16A34A", "#9333EA"]
    for i, (label, pts) in enumerate(histories.items()):
        if not pts:
            continue
        steps, vals = zip(*pts)
        ax.plot(steps, vals, label=label, color=palette[i % len(palette)], linewidth=1.8)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_lr_range_curve(csv_text: str, run_name: str) -> str:
    """Plot step vs loss from lr_range_curve.csv; return base64 PNG."""
    rows = []
    for line in csv_text.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 3:
            try:
                rows.append((int(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not rows:
        return ""
    steps, lrs, losses = zip(*rows)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twiny()
    ax1.loglog(lrs, losses, color="#2563EB", linewidth=1.8)
    ax1.set_xlabel("Learning Rate (log scale)")
    ax1.set_ylabel("Loss")
    ax2.set_xlabel("Step")
    ax2.set_xlim(0, max(steps))
    ax1.set_title(f"LR Range Test — {run_name}", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #1e293b; }
h1   { font-size: 1.8rem; border-bottom: 3px solid #2563EB; padding-bottom: 8px; }
h2   { font-size: 1.3rem; margin-top: 2.5rem; color: #1d4ed8; }
h3   { font-size: 1.05rem; color: #374151; margin-top: 1.5rem; }
table{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
th   { background: #1d4ed8; color: #fff; padding: 8px 12px; text-align: left; }
td   { padding: 7px 12px; border-bottom: 1px solid #e2e8f0; }
tr:nth-child(even) td { background: #f8fafc; }
.winner { background: #dcfce7 !important; font-weight: 600; }
.card   { border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px 20px;
          margin: 1rem 0; background: #f8fafc; }
.metric { font-size: 1.4rem; font-weight: 700; color: #1d4ed8; }
.label  { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
img     { max-width: 100%; border-radius: 6px; margin: 0.5rem 0; box-shadow: 0 1px 4px rgba(0,0,0,.1); }
.todo   { background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px;
          padding: 10px 14px; margin: 0.8rem 0; font-size: 0.9rem; }
.mono   { font-family: monospace; font-size: 0.85rem; }
.badge  { display:inline-block; padding:2px 9px; border-radius:11px; font-size:0.72rem;
          font-weight:700; text-transform:uppercase; letter-spacing:0.03em; vertical-align:middle; }
.b-done    { background:#dcfce7; color:#166534; }
.b-running { background:#dbeafe; color:#1e40af; }
.b-pending { background:#f1f5f9; color:#64748b; }
.b-blocked { background:#fee2e2; color:#991b1b; }
.b-gated   { background:#ede9fe; color:#6d28d9; }
.insight{ background:#eff6ff; border-left:4px solid #2563EB; border-radius:0 6px 6px 0;
          padding:12px 16px; margin:1rem 0; font-size:0.92rem; }
.insight strong { color:#1d4ed8; }
.pass { color:#166534; font-weight:700; } .fail { color:#94a3b8; }
.toc a { margin-right: 1rem; font-size: 0.9rem; text-decoration: none; color:#2563EB; }
.figrow { display:flex; gap:1rem; flex-wrap:wrap; align-items:flex-start; }
.figrow figure { flex:1 1 360px; margin:0.5rem 0; }
figure figcaption { font-size:0.82rem; color:#64748b; margin-top:0.2rem; }
"""


def _html_table(df: pd.DataFrame, winner_col: str | None = None,
                winner_val=None) -> str:
    if df.empty:
        return "<p><em>No data.</em></p>"
    rows_html = ""
    for _, row in df.iterrows():
        cls = ""
        if winner_col and winner_val is not None and row.get(winner_col) == winner_val:
            cls = ' class="winner"'
        cells = "".join(f"<td{cls}>{v}</td>" for v in row.values)
        rows_html += f"<tr>{cells}</tr>\n"
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{rows_html}</tbody></table>"


def _section_phase0a(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0a_arm")
    if runs.empty:
        return "<div class='todo'>⏳ Phase 0a not yet run — no data in MLflow.</div>"

    arm_rows = []
    histories = {}
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        rid = run.get("run_id", "")
        best_metric = run.get("metrics.val_realistic_pr_auc_geomean", float("nan"))
        final_loss = run.get("metrics.train_loss", float("nan"))
        val_iou = run.get("metrics.val_balanced_iou", float("nan"))
        arm_rows.append({
            "Run": name,
            "Best val_realistic_pr_auc_geomean": f"{best_metric:.4f}" if not np.isnan(best_metric) else "—",
            "Final train_loss": f"{final_loss:.4f}" if not np.isnan(final_loss) else "—",
            "Val-Balanced IoU": f"{val_iou:.4f}" if not np.isnan(val_iou) else "—",
            "Run ID": f"<span class='mono'>{rid[:8]}</span>",
        })
        hist = _get_metric_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        if hist:
            histories[name] = hist

    df = pd.DataFrame(arm_rows)
    # Find winner
    numerics = [r for r in arm_rows if r["Best val_realistic_pr_auc_geomean"] != "—"]
    winner_run = None
    if numerics:
        best = max(numerics, key=lambda r: float(r["Best val_realistic_pr_auc_geomean"]))
        winner_run = best["Run"]

    table_html = _html_table(df, winner_col="Run", winner_val=winner_run)

    chart = ""
    if histories:
        b64 = _plot_metric_curves(
            histories,
            title="Phase 0a — val_realistic_pr_auc_geomean per epoch",
            ylabel="PR-AUC geomean",
        )
        chart = f"<img src='data:image/png;base64,{b64}' alt='Phase 0a curves'>"

    winner_note = ""
    if winner_run:
        winner_note = (
            f"<div class='card'><span class='label'>Winner</span><br>"
            f"<span class='metric'>{winner_run}</span><br>"
            f"Lock this arm's normalization_stats_path into phase0b and phase0c configs.</div>"
        )

    return f"""
<h2>Phase 0a — RGB Normalization Arm Comparison</h2>
<p>Arm A: per-dataset z-score (baseline) &nbsp;|&nbsp;
   Arm B: ImageNet mean/std (/255 → imagenet) &nbsp;|&nbsp;
   Arm C: scale only (/255). Winner = Δ ≥ 0.01 over Arm A; tie-break C &gt; B.</p>
{table_html}
{winner_note}
{chart}
"""


def _section_phase0b(mlflow, experiment_name: str) -> str:
    runs_frozen = _search_runs(mlflow, experiment_name, "phase0b_lr_frozen")
    runs_unfrozen = _search_runs(mlflow, experiment_name, "phase0b_lr_unfrozen")

    sections = []
    for label, runs in [("Frozen backbone", runs_frozen), ("Unfrozen (full fine-tune)", runs_unfrozen)]:
        if runs.empty:
            sections.append(f"<div class='todo'>⏳ {label} LR range test not yet run.</div>")
            continue
        run = runs.iloc[0]
        rid = run.get("run_id", "")
        csv_text = _download_artifact_text(mlflow, rid, "lr_range_curve.csv")
        if csv_text:
            b64 = _plot_lr_range_curve(csv_text, run.get("tags.mlflow.runName", label))
            sections.append(f"<h3>{label}</h3><img src='data:image/png;base64,{b64}' alt='LR range {label}'>")
        else:
            sections.append(f"<div class='todo'>⏳ {label}: lr_range_curve.csv not found in MLflow artifacts.</div>")

    return f"""
<h2>Phase 0b — LR Range Test</h2>
<p>Pick <code>frozen_lr</code> (frozen run) and <code>base_lr</code> (unfrozen run) at the steepest
stable loss descent before divergence. Update phase0b and phase0c configs before running 0c.</p>
{"".join(sections)}
"""


def _fmt(v: float, places: int = 4) -> str:
    """Format a metric value, or em-dash for NaN."""
    return f"{v:.{places}f}" if not np.isnan(v) else "—"


def _best_smoothed_from_history(mlflow, run_id: str, metric: str, window: int = 3) -> float:
    """Max of the trailing `window`-validation moving average.

    Replicates training/early_stopping.py so the reported "best" matches the
    smoothed value the early-stopper used for best-checkpoint selection
    (window = training.early_stopping.smoothing_window, 3 in all phase0 configs).
    """
    from collections import deque
    vals = [v for _, v in sorted(_get_metric_history(mlflow, run_id, metric))]
    if not vals:
        return float("nan")
    win: deque = deque(maxlen=window)
    best = float("-inf")
    for v in vals:
        win.append(v)
        best = max(best, sum(win) / len(win))
    return best


def _final_from_history(mlflow, run_id: str, metric: str) -> float:
    """Last logged value of a metric (NaN if none)."""
    hist = _get_metric_history(mlflow, run_id, metric)
    return hist[-1][1] if hist else float("nan")


def _section_phase0c(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0c_seed")
    if runs.empty:
        return "<div class='todo'>⏳ Phase 0c multi-seed baseline not yet run.</div>"

    # Defensive: a relaunched seed creates a second run with the same name.
    # Keep only the most-recent run per seed so μ₀/σ₀ aren't double-counted.
    if {"tags.mlflow.runName", "start_time"}.issubset(runs.columns):
        runs = (runs.sort_values("start_time")
                    .drop_duplicates("tags.mlflow.runName", keep="last"))

    # Gate metric is the geomean over the honestly-supported ratios [5,10,20]
    # (metrics.pr_auc_ratios; see docs/baseline_unetpp_effb5.md). pixel_iou and
    # obj_f1 are logged as monotonic stability anchors.
    seed_rows = []
    best_vals = []
    curve_blocks = []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        rid = run.get("run_id", "")
        # Best-per-seed = MAX over the epoch history (project defines μ₀/σ₀ on
        # the best-per-seed gate value, not the last-logged one).
        best = _best_smoothed_from_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        seed_rows.append({
            "Run": name,
            "PR-AUC geomean (best)": f"{best:.4f}" if not np.isnan(best) else "—",
            "PR-AUC 1:5 (final)": _fmt(_final_from_history(mlflow, rid, "pr_auc_ratio_5")),
            "PR-AUC 1:10 (final)": _fmt(_final_from_history(mlflow, rid, "pr_auc_ratio_10")),
            "PR-AUC 1:20 (final)": _fmt(_final_from_history(mlflow, rid, "pr_auc_ratio_20")),
            "pixel_IoU (final)": _fmt(_final_from_history(mlflow, rid, "pixel_iou")),
            "obj_F1 (final)": _fmt(_final_from_history(mlflow, rid, "object_f1")),
        })
        # μ₀/σ₀ are calibrated on COMPLETED seeds only — a still-running seed's
        # best-so-far would otherwise contaminate the gate.
        if not np.isnan(best) and run.get("status", "") == "FINISHED":
            best_vals.append(best)

        # Per-seed curve panels: (1) train vs val loss overlay — overfitting
        # detector; (2) gate metric + IoU/F1 quality anchors.
        loss_hist = {
            "train_loss": _get_metric_history(mlflow, rid, "train_loss"),
            "val_loss": _get_metric_history(mlflow, rid, "val_loss"),
        }
        qual_hist = {
            "PR-AUC geomean (gate)": _get_metric_history(mlflow, rid, "val_realistic_pr_auc_geomean"),
            "pixel_IoU": _get_metric_history(mlflow, rid, "pixel_iou"),
            "obj_F1": _get_metric_history(mlflow, rid, "object_f1"),
        }
        imgs = []
        if any(loss_hist.values()):
            b64 = _plot_metric_curves(loss_hist, title=f"{name} — train vs val loss",
                                      ylabel="loss", colors=["#2563EB", "#DC2626"])
            imgs.append(f"<img src='data:image/png;base64,{b64}' alt='{name} loss'>")
        if any(qual_hist.values()):
            b64 = _plot_metric_curves(qual_hist, title=f"{name} — gate metric + quality anchors",
                                      ylabel="score", colors=["#9333EA", "#16A34A", "#EA580C"])
            imgs.append(f"<img src='data:image/png;base64,{b64}' alt='{name} quality'>")
        if imgs:
            curve_blocks.append(
                f"<div style='display:flex; gap:1rem; flex-wrap:wrap; margin:0.5rem 0;'>"
                f"{''.join(imgs)}</div>")

    df = pd.DataFrame(seed_rows)
    table_html = _html_table(df)

    stats_html = ""
    if len(best_vals) >= 2:
        n_done = len(best_vals)
        prelim = ("<p style='color:#B45309; font-weight:600;'>⚠ Preliminary — "
                  f"only {n_done}/3 seeds finished; σ₀ from &lt;3 seeds is unreliable. "
                  "Final gate requires all 3.</p>") if n_done < 3 else ""
        mu0 = float(np.mean(best_vals))
        sigma0 = float(np.std(best_vals, ddof=1))
        # experiments.md §1.4: a candidate wins iff Δ(PR-AUC geomean) vs baseline μ₀ ≥ G
        # AND precision@recall=0.5 does not regress. G is a Δ-threshold, NOT a perf floor.
        gate_g = max(0.01, 2 * sigma0)
        if sigma0 < 0.005:
            band = "Low-noise (σ₀ < 0.005) — single seed per candidate reliable"
        elif sigma0 < 0.015:
            band = "Medium-noise (0.005 ≤ σ₀ < 0.015) — single-seed first-pass; re-run top ties at seed 43"
        else:
            band = "High-noise (σ₀ ≥ 0.015) — investigate noise before continuing"

        stats_html = f"""
<div class='card'>
  {prelim}
  <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
    <div><span class='label'>μ₀ (mean best PR-AUC geomean — baseline ref)</span><br><span class='metric'>{mu0:.4f}</span></div>
    <div><span class='label'>σ₀ (std across seeds)</span><br><span class='metric'>{sigma0:.4f}</span></div>
    <div><span class='label'>Gate G = max(2σ₀, 0.01)</span><br><span class='metric'>{gate_g:.4f}</span></div>
  </div>
  <p style='margin-top:0.8rem; font-size:0.9rem;'><strong>Noise band:</strong> {band}</p>
  <p style='margin-top:0.4rem; font-size:0.85rem; color:#555;'>Gate metric = geomean(PR-AUC @ 1:5/1:10/1:20).
  Per <code>experiments.md §1.4</code>, a candidate <strong>wins</strong> only if Δ(PR-AUC geomean) vs
  baseline μ₀={mu0:.4f} is ≥ <strong>G = {gate_g:.4f}</strong> <em>and</em> precision @ recall=0.5 does not regress.</p>
</div>"""

    return f"""
<h2>Phase 0c — Multi-Seed Baseline</h2>
<p>Seeds 42, 43, 44 on the frozen dataset snapshot with locked normalization and LRs from Phase 0a/0b.
   μ₀ is the baseline; σ₀ sets the winner gate <strong>G = max(2σ₀, 0.01)</strong> (experiments.md §1.4).</p>
{table_html}
{stats_html}
<h3>Per-seed training curves</h3>
<p>Left: train vs val loss (overfitting detector). Right: gate metric with pixel_IoU / obj_F1 anchors.</p>
{"".join(curve_blocks) if curve_blocks else "<p>(no curve history available)</p>"}
"""


def _section_artifacts(mlflow, experiment_name: str) -> str:
    runs = _search_runs(mlflow, experiment_name, "phase0")
    if runs.empty:
        return "<div class='todo'>⏳ No Phase 0 runs found in MLflow.</div>"

    rows = []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", "?")
        rid = run.get("run_id", "")
        tracking_uri = run.get("artifact_uri", "")
        status = run.get("status", "")
        rows.append({
            "Run name": name,
            "Run ID": f"<span class='mono'>{rid}</span>",
            "Status": status,
            "Artifact URI": f"<span class='mono' style='font-size:0.8rem'>{tracking_uri}</span>",
            "Checkpoints": f"<span class='mono' style='font-size:0.8rem'>runs/{name}/checkpoints/</span>",
        })
    df = pd.DataFrame(rows)
    return f"""
<h2>Artifact Locations</h2>
<p>Every training run emits an artifact summary to the log. Checkpoints are saved to
<code>runs/&lt;run_name&gt;/checkpoints/</code> on the host VM and to MLflow as
<code>best_deployment.pth</code> at run end. MLflow artifacts are at the Tracking URI below.</p>
{_html_table(df)}
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Project-wide sections (experiments.md program: Phase 0 → 5 → Final)
# ---------------------------------------------------------------------------

# v1.0 locked-baseline constants (docs/phase0_baseline.md, 2026-06-14). Locked for
# the project; _section_phase0c also computes the live values from MLflow, these
# mirror them for the cross-section gate comparisons.
MU0, SIGMA0 = 0.791174, 0.005587
GATE_G = max(0.01, 2 * SIGMA0)        # 0.0112 — experiments.md §1.4 (winner bar μ₀+G=0.8023)
GATE_RATIOS = "[5, 10, 20]"
PCT_TO_NPOS = {25: 328, 50: 656, 75: 983, 100: 1311}   # v1.0 train positives = 1311

# v2 deploy reference (corrected leakage-free split). 3-seed mean of the locked recipe
# (RGB+NDVI · F0 · focal·ignore_w2 · default sampling · aug−RandomScale · base_v2_fast).
# SSoT: docs/experiment_ledger.md row #64. Corrected-split σ ≈ 0.012 (~2× the leaky σ₀).
DEPLOY_MEAN = 0.9123
DEPLOY_SEEDS = (0.9144, 0.9068, 0.9156)
SIGMA_CORRECTED = 0.012


# ---------------------------------------------------------------------------
# Curated, ledger-sourced content (SSoT = docs/experiment_ledger.md).
# These mirror the ledger the same way MU0/SIGMA0 mirror docs/phase0_baseline.md.
# Update here whenever the ledger's verdicts change. Family scheme A–K.
# ---------------------------------------------------------------------------

# Per-family "what we learned" — the experiment-group narrative. Each entry:
#   id, name, status (badge), learned (the headline insight), evidence (deciding runs/numbers).
FAMILY_LEARNINGS = [
    dict(id="A", name="Baseline & gate", status="done",
         learned="A reproducible baseline and an honest winner gate were established before any "
                 "experiment was trusted. Everything downstream is measured against this.",
         evidence="3-seed baseline μ₀=0.7912 (leaky split); seed std σ₀=0.0056 → gate "
                  "G=max(2σ₀,0.01)=0.0112. Later re-measured on the corrected split: σ≈0.012 "
                  "(~2× larger) → a LOCK now needs mean Δ≥G <em>and</em> sign-consistency across 3 seeds."),
    dict(id="B", name="Data scaling", status="done",
         learned="More labeled data is NOT the near-term lever — v1.0 is on a data plateau and the "
                 "model is well-matched to its data (not over-parameterized). This is the first pillar "
                 "of the central diagnosis: we are representation-limited, not data-volume-limited.",
         evidence="Data-scaling slope (75→100)/(25→50) is flat; train/val IoU gap ≈0.05 best / 0.17 "
                  "final (≪0.4). 25%→100%: 0.764→0.790 (leaky). No new labels are coming, so the fixed "
                  "1.5k positives must be squeezed by representation, not volume."),
    dict(id="C", name="Loss & boundary", status="done",
         learned="The win is the BOUNDARY treatment, not the loss function. Focal alone is the loss "
                 "winner; precision-skewed Tversky collapses the imbalanced gate. Adding a boundary-"
                 "ignore band is what clears the gate — both loss families pass with it, neither without.",
         evidence="focal·ignore_w2 = 0.805 (leaky; seed-confirmed 0.805–0.820), tied with compound "
                  "1:2·ignore_w3 (0.802–0.817); chosen for simplicity. Tversky 2:8 collapsed (0.073). "
                  "LOCKED: focal + boundary-ignore width 2."),
    dict(id="D", name="Channels & fusion", status="done",
         learned="A single well-chosen extra channel (NDVI) is the biggest representation win — and "
                 "more is not better. Adding further channels or heavier fusion architectures extracts "
                 "LESS than the simple 4-channel early-stack. The plateau is about signal, not plumbing.",
         evidence="NDVI-alone 3-seed mean 0.8985 ≫ RGB control 0.830 (+0.07, ≫σ) and > full 8-band "
                  "0.869. Greedy forward from NDVI added nothing (+se_pca 0.900, +tc 0.900, +se_proto "
                  "0.898, +nbr 0.856 — none clears G). Heavy fusion LOSES: F3-full ~0.78, F5-full ~0.83, "
                  "F5-pair ~0.87 (all ≪ NDVI-alone). LOCKED: EXTRA=[NDVI], F0 early channel-stack."),
    dict(id="E", name="Architecture & encoder", status="running",
         learned="Capacity is not the lever (consistent with B/F). No CNN decoder beats the dense-skip "
                 "UNet++, and scaling capacity down (EffB3) or up doesn't help. The open bet is the "
                 "ENCODER representation: generic web-DINOv3 helps on RGB but its edge vanishes once NDVI "
                 "is added; a satellite-pretrained encoder is the live candidate.",
         evidence="UNet++/EffB5 ≥ FPN (0.794 tie) > DeepLabV3+ > PSPNet ≫ MANet (0.621). EffB3 0.9050 "
                  "(Δ−0.007, no-win). web-DINOv3+NDVI ties EffB5+NDVI (~0.912) → generic foundation is "
                  "not the lever. <strong>Satellite DINOv3 ViT-L (SAT-493M) fine-tuned reached 0.9187 @ "
                  "ep30 (climbing)</strong>; the frozen 7B probe was non-competitive (best 0.498) and "
                  "diverged (constant frozen_lr) → killed. SAM2/Hiera RGB non-competitive (0.59)."),
    dict(id="F", name="Augmentation", status="running",
         learned="Augmentation is NOT the plateau-breaker (the third pillar of the representation-limited "
                 "diagnosis). Photometric aug genuinely helps and must be kept; downscale (RandomScale) "
                 "HURTS; and the whole sample-mixing family fails to beat the deploy recipe. Auto-policies "
                 "are the last aug angle being tested.",
         evidence="Geometric-only craters to 0.794 (−0.072) → photometric matters; CLAHE/×1.5 within "
                  "noise → keep, don't strengthen. Drop-RandomScale +0.016, positive in 3/3 seeds → "
                  "LOCKED drop. Mixing augs all no-win vs deploy 0.9123: copy-paste 0.893 (worst — breaks "
                  "spatial-context/shadow cues), cutmix 0.901, mosaic 0.907, mixup ~. RandAugment / "
                  "TrivialAugment (shadow-safe pool) running."),
    dict(id="G", name="Sampling / curriculum", status="done",
         learned="Default balanced sampling is sufficient — the curriculum 'win' did not survive seeds.",
         evidence="Curriculum r20_pf33: 0.894/0.901/0.859 → mean ≈0.885 vs base 0.879 (Δ≈0.006, sign "
                  "flipped on seed 44) → within noise, REJECTED. Default sampling LOCKED."),
    dict(id="H", name="Calibration & TTA", status="pending",
         learned="Not yet run — the inference-time squeeze (temperature + threshold + D4-TTA) is held "
                 "for the final lock. Scale-TTA is gated on a separate scale-transfer test.",
         evidence="Required before ship; adopt D4-TTA if ≥1% PR-AUC at ≤0.5% precision cost."),
    dict(id="I", name="Final lock & test", status="done",
         learned="The locked v2 recipe's components are additive: boundary-ignore + drop-RandomScale "
                 "stack cleanly on top of NDVI for the strongest model so far. Test-Realistic is touched "
                 "exactly once, at the very end.",
         evidence="Deploy 3-seed 0.9144/0.9068/0.9156 → mean <strong>0.9123</strong> (spread 0.907–0.916), "
                  "+0.014 over NDVI-alone 0.8985. Re-locks only if a pre-ship screen earns it."),
    dict(id="J", name="Deploy & inference", status="pending",
         learned="Deployment target is decided (L4 fleet for the 41.5M-tile pan-Arctic pass); the run "
                 "happens after the winner locks. Pan-Arctic mapping reads as a QC-assisted candidate map "
                 "(high recall + filtering), not yet a fully-automated high-precision product.",
         evidence="us-west1 L4 fleet decided; inference pipeline drafted (PR #19/#23)."),
    dict(id="K", name="Deferred / discussed", status="pending",
         learned="Several deeper bets are deliberately deferred to v3 or gated, so they never block the "
                 "v2 ship: re-stage (+28 pos), hard-negative mining (post first inference), MAE SSL "
                 "pretraining (end-stage), pseudo-labeling (backup only).",
         evidence="See the ledger's dropped/discussed table for the full record + the reason each idea "
                  "isn't in v2 (e.g. SegFormer/EffB7/UNet3+/YOLO/SAM3 all dropped with cause)."),
]

# The meta-learnings that cut across families — the project's central thesis.
CROSS_CUTTING = [
    ("Representation-limited, not capacity / volume / regularization-limited",
     "The single most important finding, triangulated from three independent families: data scaling is "
     "a plateau with a well-matched model (B), no bigger/smaller backbone or decoder helps (E), and "
     "neither heavier regularization nor more augmentation helps (F). So leverage lives in richer "
     "representation (channels, encoders, SSL) — not in scale or regularization."),
    ("NDVI is the efficient representation lever — and more is not better",
     "One channel (NDVI) delivers the bulk of the gain over RGB (+0.07); adding channels, channel-"
     "attention, or dual-encoder/cross-modal fusion all extract LESS (D). The signal ceiling is reached "
     "by a simple 4-channel early-stack."),
    ("The boundary, the downscale, and photometric aug are the cheap, real wins",
     "Three components stack additively onto NDVI for the deploy recipe: boundary-ignore_w2 (C), "
     "dropping RandomScale downscale (F, +0.016/3-of-3), and keeping the photometric set (F, geometric-"
     "only is −0.072)."),
    ("Foundation encoders: generic helps RGB, but satellite-pretraining is the live bet",
     "web-DINOv3 beats EffB5 on RGB (+0.043) but the edge vanishes once NDVI is added (ties at ~0.912). "
     "The remaining encoder bet is a satellite-domain model — sat-DINOv3 ViT-L fine-tuned is at 0.92 and "
     "climbing (E)."),
]

# Locked decisions — every locked choice + how it was decided.
LOCKED_DECISIONS = [
    ("Normalization", "Per-dataset z-score (Arm A)", "A",
     "Arm A 0.667 vs B 0.626 / C 0.670 (tie, A kept as default); locked into all phase0c+ configs."),
    ("Loss", "Focal", "C",
     "Focal beats Tversky (collapses) and compound 1:2 (near-miss) on the imbalanced gate."),
    ("Boundary", "Ignore band, width 2", "C",
     "focal·ignore_w2 0.805 (seed-confirmed 0.805–0.820); the gate-clearing win. Tied w/ compound·w3, "
     "kept for simplicity (single loss + narrower discarded band)."),
    ("Decoder / backbone", "UNet++ / EfficientNet-B5", "E",
     "No smp decoder beats it (FPN ties, MANet collapses); EffB3 capacity-down is −0.007."),
    ("EXTRA channels", "NDVI only (4-ch RGB+NDVI)", "D",
     "NDVI 3-seed 0.8985 ≫ RGB 0.830 and > full 8-band 0.869; greedy forward added no channel (all <G)."),
    ("Fusion", "F0 early channel-stack", "D",
     "F0/F1/F2 tie; heavy F3/F5 lose (≪ NDVI-alone) → simplest fusion locked, evidence-based."),
    ("Augmentation — scale", "Drop RandomScale downscale", "F",
     "3-seed A/B +0.016, positive in all 3 seeds (sign-consistent)."),
    ("Augmentation — photometric", "Keep photometric set + CLAHE", "F",
     "Geometric-only is −0.072; dropping CLAHE / ×1.5 are within noise → keep as-is."),
    ("Sampling", "Default balanced (no curriculum)", "G",
     "Curriculum r20_pf33 mean Δ≈0.006 with a flipped seed → within noise, rejected."),
    ("Stop schedule", "base_v2_fast (patience 5, start 45, max 120)", "—",
     "Gate-neutral (fastcheck 0.8934 ≈ original 0.888); ~2× throughput, best checkpoint unchanged."),
    ("Deploy recipe (v2)", "RGB+NDVI · F0 · focal·ignore_w2 · default · −RandomScale", "I",
     "3-seed mean 0.9123 (0.9144/0.9068/0.9156); re-locks only if a pre-ship screen earns it."),
]


def _dedup_latest(runs):
    """Keep the most-recent run per run_name (relaunches create duplicates)."""
    if not runs.empty and {"tags.mlflow.runName", "start_time"}.issubset(runs.columns):
        return runs.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
    return runs


def _badge(status: str) -> str:
    cls = {"done": "b-done", "running": "b-running", "pending": "b-pending",
           "blocked": "b-blocked", "gated": "b-gated"}.get(status, "b-pending")
    return f"<span class='badge {cls}'>{status}</span>"


def _plot_data_scaling(points) -> str:
    """points = [(n_pos, gate)]; gate vs log10(n_pos)."""
    pts = sorted(points)
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, "o-", color="#2563EB", linewidth=2, markersize=8)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 9), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("train positives (log scale)")
    ax.set_ylabel("best gate (PR-AUC geomean)")
    ax.set_title("Phase 2 — data-scaling curve")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    return _fig_to_base64(fig)


def _plot_gap_bars(labels, train_ious, val_ious) -> str:
    """Grouped bars: train vs val pixel-IoU per subset (generalization gap)."""
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, train_ious, w, label="train IoU", color="#2563EB")
    ax.bar(x + w / 2, val_ious, w, label="val IoU", color="#DC2626")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("pixel IoU (final epoch)")
    ax.set_title("Phase 2 §5.4 — train vs val IoU (gap = overfitting)")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _section_overview(tracking_uri: str) -> str:
    # Dataset facts come from the SSoT (data/version.json), never hardcoded.
    try:
        vj = json.loads(Path("data/version.json").read_text())
        ds = f"v{vj['version']}"
        tc = vj["tile_counts"]
        tiles = f"{tc['total']:,} tiles · {tc['positive']:,} pos"
    except Exception:
        ds, tiles = "v1.0", ""
    return f"""
<h2 id='overview'>1. Overview &amp; status</h2>
<p>Semantic segmentation of <strong>Retrogressive Thaw Slumps (RTS)</strong> in pan-arctic PlanetScope
imagery (60–74°N). UNet++ / EfficientNet-B5; balanced sampling + curriculum + focal loss. This is the
project's single living dashboard — auto-generated from MLflow (<code>{tracking_uri}</code>).</p>
<div class='card'>
  <div style='display:flex; gap:2rem; flex-wrap:wrap;'>
    <div><span class='label'>Dataset</span><br><span class='metric'>{ds}</span> <span style='font-size:0.8rem'>{tiles}</span></div>
    <div><span class='label'>v2 deploy (corrected, 3-seed)</span><br><span class='metric'>{DEPLOY_MEAN:.4f}</span></div>
    <div><span class='label'>Baseline μ₀ (leaky)</span><br><span class='metric'>{MU0:.4f}</span></div>
    <div><span class='label'>Gate G = max(2σ₀,0.01)</span><br><span class='metric'>{GATE_G:.3f}</span></div>
    <div><span class='label'>Stage</span><br><span class='metric'>Pre-ship screens</span></div>
  </div>
  <p style='margin-top:0.6rem; font-size:0.85rem; color:#555'>Model = <strong>v2</strong>. A candidate
  <strong>wins</strong> iff Δ(PR-AUC geomean) vs the reference ≥ G <em>and</em> precision@recall0.5 does
  not regress (experiments.md §1.4); every LOCK also requires 3-seed sign-consistency. Gate metric =
  geomean(PR-AUC @ {GATE_RATIOS}); 1:200/1000 deferred to Test-Realistic. Current reference for new
  screens is the <strong>v2 deploy {DEPLOY_MEAN:.4f}</strong> (corrected split); μ₀ is the original
  leaky-split anchor for the gate width.</p>
</div>
<p class='toc'><strong>Jump:</strong>
<a href='#learnings'>What we learned</a><a href='#locked'>Locked decisions</a><a href='#dashboard'>Overview</a>
<a href='#p0a'>Phase 0</a><a href='#p2'>Phase 2</a><a href='#p3'>Phase 3</a>
<a href='#p4'>Phase 4</a><a href='#p5'>Phase 5</a><a href='#findings'>Findings</a><a href='#future'>Future</a></p>
"""


def _section_phase1() -> str:
    return f"""
<h2 id='p1'>3. Phase 1 — Temporal sanity (2025) {_badge('blocked')}</h2>
<p>Detect 2024→2025 domain drift on a 2025 micro-set (experiments.md §4) — <strong>blocked</strong> on the
micro-set definition (user + Heidi Rodenhizer). ~1 GPU-hour once it exists.</p>
"""


def _section_phase2(mlflow, experiment_name: str) -> str:
    runs = _dedup_latest(_search_runs(mlflow, experiment_name, "phase2_scale_"))
    if runs.empty:
        return f"<h2 id='p2'>4. Phase 2 — Data scaling {_badge('pending')}</h2><p>experiments.md §5.</p>"
    rows, curve, glab, gtr, gva = [], [], [], [], []
    for _, run in runs.iterrows():
        name = run.get("tags.mlflow.runName", ""); rid = run.get("run_id", "")
        try:
            pct = int(name.split("_")[-1])
        except ValueError:
            continue
        gate = _best_smoothed_from_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        tiou = _final_from_history(mlflow, rid, "train_iou")
        viou = _final_from_history(mlflow, rid, "pixel_iou")
        n = PCT_TO_NPOS.get(pct, pct)
        curve.append((n, gate))
        gap = (tiou - viou) if not (np.isnan(tiou) or np.isnan(viou)) else float("nan")
        rows.append({"Subset": f"{pct}%", "≈ pos": n, "Best gate": _fmt(gate),
                     "train IoU": _fmt(tiou, 3), "val IoU": _fmt(viou, 3), "gap": _fmt(gap, 3)})
        if not (np.isnan(tiou) or np.isnan(viou)):  # skip subsets without both IoUs (a 0.0 bar fakes a negative gap)
            glab.append(f"{pct}%"); gtr.append(tiou); gva.append(viou)
    base = _dedup_latest(_search_runs(mlflow, experiment_name, "phase0c_seed42"))
    if not base.empty:
        g100 = _best_smoothed_from_history(mlflow, base.iloc[-1]["run_id"], "val_realistic_pr_auc_geomean")
        curve.append((1900, g100))
        rows.append({"Subset": "100%", "≈ pos": 1900, "Best gate": _fmt(g100),
                     "train IoU": "—", "val IoU": "—", "gap": "—"})
    rows = sorted(rows, key=lambda r: r["≈ pos"])
    cmap = {n: g for n, g in curve}
    ratio_txt, regime = "—", "—"
    try:
        s_lo = (cmap[950] - cmap[475]) / (np.log(950) - np.log(475))
        s_hi = (cmap[1900] - cmap[1425]) / (np.log(1900) - np.log(1425))
        ratio = s_hi / s_lo if s_lo != 0 else float("inf")
        if np.isnan(ratio):  # a run mid-training yields NaN gates; NaN compares
            raise KeyError   # False everywhere and would misreport "Plateau"
        ratio_txt = f"{ratio:.1f}"
        regime = ("Severely under-scaled" if ratio > 1.0 else
                  "Diminishing but still scaling" if ratio >= 0.5 else "Plateau before 100%")
    except KeyError:
        pass
    imgs = ""
    if len(curve) >= 2:
        imgs += f"<img src='data:image/png;base64,{_plot_data_scaling(curve)}'>"
    if glab:
        imgs += f"<img src='data:image/png;base64,{_plot_gap_bars(glab, gtr, gva)}'>"
    badge = _badge("done") if len(rows) >= 4 else _badge("running")
    return f"""
<h2 id='p2'>4. Phase 2 — Data scaling {badge}</h2>
<p>Does more labeled data help, and is the model using its capacity? (experiments.md §5)</p>
<div style='display:flex; gap:1rem; flex-wrap:wrap;'>{imgs}</div>
{_html_table(pd.DataFrame(rows))}
<div class='insight'><strong>§5.3 slope</strong> (75→100)/(25→50) ≈ <strong>{ratio_txt}</strong> →
<strong>{regime}</strong>.<br>
<strong>§5.4 gap</strong>: see table — on v1.0 the train/val IoU gap is small (≈0.05 best / 0.17 final,
&lt; 0.4) → the model is <strong>well-matched to its data, not over-parameterized</strong>. Combined with the
flat slope this reads as a <strong>data/representation plateau</strong>, not a pure data-volume or capacity
limit, so leverage pivots to <strong>representation (Phase 4 EXTRA channels)</strong> rather than bigger
backbones or heavier regularization (experiments.md §1.6).</div>
"""


def _phase3_rows(mlflow, experiment_name: str, *prefixes: str) -> pd.DataFrame:
    """Δ-vs-μ₀ gate table for the given run-name prefixes (live from MLflow)."""
    frames = [_dedup_latest(_search_runs(mlflow, experiment_name, p)) for p in prefixes]
    allruns = pd.concat([f for f in frames if not f.empty]) if any(not f.empty for f in frames) else frames[0]
    rows = []
    for _, run in allruns.iterrows():
        name = run.get("tags.mlflow.runName", ""); rid = run.get("run_id", "")
        gate = _best_smoothed_from_history(mlflow, rid, "val_realistic_pr_auc_geomean")
        if np.isnan(gate):
            continue
        d = gate - MU0
        passed = d >= GATE_G
        label = name
        for p in prefixes:
            label = label.replace(p, "")
        rows.append({"Candidate": label, "Best gate": _fmt(gate), "Δ vs baseline": f"{d:+.4f}",
                     "Win (≥G)?": f"<span class='{'pass' if passed else 'fail'}'>{'PASS' if passed else 'no'}</span>"})
    return pd.DataFrame(rows)


def _section_phase3(mlflow, experiment_name: str) -> str:
    # §1.4 defines the win as Δ vs baseline μ₀ (multi-seed mean), not the single seed-42 number.
    ref = pd.DataFrame([{"Candidate": "focal (baseline μ₀)", "Best gate": _fmt(MU0),
                         "Δ vs baseline": "ref", "Win (≥G)?": "—"}])
    loss = _phase3_rows(mlflow, experiment_name, "phase3_loss_", "abl_loss_")
    boundary = _phase3_rows(mlflow, experiment_name, "phase3_bd_")
    loss_tbl = _html_table(pd.concat([ref, loss], ignore_index=True))
    bd_tbl = _html_table(pd.concat([ref, boundary], ignore_index=True)) if not boundary.empty \
        else "<p><em>No boundary runs yet.</em></p>"
    return f"""
<h2 id='p3'>5. Phase 3 — Loss family → boundary {_badge('done')}</h2>
<p>Sequential elimination (experiments.md §6): pick loss, lock, then boundary. Win = Δ ≥ G={GATE_G:.3f} + no precision drop.</p>
<h3>5a. Loss family</h3>
{loss_tbl}
<h3>5b. Boundary factorial (focal &amp; compound 1:2 × ignore-width)</h3>
{bd_tbl}
<div class='insight'><strong>v1.0 read (live from MLflow above):</strong> no loss beat the gate →
<strong>focal stays the loss winner</strong> (tversky collapses the imbalanced gate; compound 1:2 is a
near-miss). The boundary factorial is the first real win: <strong>boundary winner LOCKED = focal + ignore-width 2</strong>
(3-seed mean 0.8100, clears the bar across all seeds), statistically tied with compound 1:2 + width 3
(0.8116) and chosen for simplicity. <strong>The win is the boundary, not the loss</strong> — both loss
families clear the gate once <code>ignore</code> is added, neither without. Deploy architecture so far:
<strong>UNet++/EffB5 + focal + ignore_w2</strong>. Caveat: leaky-split numbers (absolute optimistic, relative
preserved); the corrected split is used for Phase-4 + the final test. Full analysis: <code>docs/phase3_loss_boundary.md</code>.</div>
"""


def _extra_vis_block() -> str:
    """Embed one positive (with RTS label contour) + one negative EXTRA-vis panel."""
    import glob
    pos = sorted(glob.glob(f"{_EXTRA_VIS}/pos_*.png")) or \
        sorted(glob.glob(f"{_EXTRA_VIS.replace('/outputs/', '/mnt/outputs/')}/pos_*.png"))
    neg = sorted(glob.glob(f"{_EXTRA_VIS}/neg_*.png")) or \
        sorted(glob.glob(f"{_EXTRA_VIS.replace('/outputs/', '/mnt/outputs/')}/neg_*.png"))
    out = []
    if pos and (b := _img_file_b64(pos[0])):
        out.append(f"<figure><img src='data:image/png;base64,{b}' alt='EXTRA positive'>"
                   f"<figcaption>Positive tile — RGB + 8 EXTRA bands (RTS label contour overlaid).</figcaption></figure>")
    if neg and (b := _img_file_b64(neg[0])):
        out.append(f"<figure><img src='data:image/png;base64,{b}' alt='EXTRA negative'>"
                   f"<figcaption>Negative tile — featureless by comparison.</figcaption></figure>")
    return "".join(out) if out else "<p><em>EXTRA-vis panels not found.</em></p>"


def _section_phase45(mlflow, experiment_name: str) -> str:
    arch = _phase3_rows(mlflow, experiment_name, "phase5_arch_")
    ref = pd.DataFrame([{"Candidate": "UNet++/EffB5 (baseline μ₀)", "Best gate": _fmt(MU0),
                         "Δ vs baseline": "ref", "Win (≥G)?": "—"}])
    arch_tbl = _html_table(pd.concat([ref, arch], ignore_index=True)) if not arch.empty \
        else "<p><em>No architecture runs yet.</em></p>"
    return f"""
<h2 id='p4'>6. Phase 4 — EXTRA channels {_badge('running')}</h2>
<p>EXTRA group ablation — NDVI / NBR / SE-PCA / SE-Proto / TC (experiments.md §7), the primary
plateau-breaker (§1.6). <strong>Full 8-band stack now generated for all 22,259 tiles</strong> (S2 bands
NDVI/NBR/TC + SE bands: global-PCA(3) and a contrastive RTS prototype), with per-channel normalization
(data.md §9). The <strong>ablation wave is running on the corrected leakage-free split</strong> (RGB
control + 5 single-group + full-stack), inheriting the focal baseline (boundary held at none — the
boundary win is additive). A NaN-fill fix (<code>apply_norm</code> neutralizes EXTRA NoData where the
source has no coverage) was required to clear a first-validation crash. Quantitative results pending
(early epochs; corrected-split absolutes are not comparable to the leaky-split μ₀ above).</p>
<p><strong>EXTRA carries RTS-specific signal</strong> (the accept gate before trusting the stack):</p>
<div class='figrow'>{_extra_vis_block()}</div>
<h2 id='p5'>7. Phase 5 — Architecture {_badge('running')}</h2>
<p>Run-now architecture sweep (experiments.md §8): smp decoder drop-ins on EffB5 (frozen HP, §8.2);
encoder-family swaps (SegFormer/DINOv3) need per-family HP retuning (§8.2a) before a fair comparison.</p>
{arch_tbl}
<div class='insight'><strong>v1.0 read (live above):</strong> decoder swaps so far underperform the
UNet++ baseline (FPN ≈ μ₀, MAnet collapses); the dense-skip UNet++ remains the architecture to beat.
Phase 5 is exploratory here (not the primary lever — see §1.6: the v1.0 plateau points to representation,
not capacity).</div>
"""


def _section_findings() -> str:
    ov = _img_file_b64(_VAL_OVERLAY)
    ov24 = _img_file_b64(_VAL_OVERLAY_2024)
    val_figs = []
    if ov24:
        val_figs.append(f"<figure><img src='data:image/png;base64,{ov24}' alt='2024-label overlay'>"
                        f"<figcaption>Baseline probability vs <strong>2024 RTS labels</strong> (Banks Is. AOI): "
                        f"mean prob <strong>0.578 inside labels vs 0.071 outside (8×)</strong>.</figcaption></figure>")
    if ov:
        val_figs.append(f"<figure><img src='data:image/png;base64,{ov}' alt='2025 inference overlay'>"
                        f"<figcaption>Merged 2025 inference probability over the validation AOI.</figcaption></figure>")
    val_block = (f"<h3>Baseline qualitative validation</h3><div class='figrow'>{''.join(val_figs)}</div>"
                 if val_figs else "")
    return f"""
<h2 id='findings'>8. Findings &amp; insights (v1.0)</h2>
{val_block}
<ul>
<li><strong>Representation, not capacity or volume, is the bottleneck.</strong> v1.0 data-scaling is a
<strong>plateau</strong> (flat slope) and the model is <strong>well-matched</strong> (train/val IoU gap
≈0.05 best / 0.17 final, &lt; 0.4) → bigger backbones and heavier regularization are low-leverage; the
plateau-breaker is <strong>EXTRA channels (Phase 4)</strong>.</li>
<li><strong>Focal-only loss wins</strong>; precision-focused Tversky collapses the imbalanced gate; compound
(Focal+Dice) is a near-miss.</li>
<li><strong>Boundary-ignore is the win</strong> — boundary winner <strong>locked = focal + ignore-width 2</strong>
(3-seed mean 0.8100), tied with compound 1:2 + width 3. The win is the boundary, not the loss; both losses
clear the gate once <code>ignore</code> is added, neither without.</li>
<li><strong>Gate is honest-ratio</strong> — measured at 1:5–1:20, not deployment 1:1000, so these numbers
are optimistic vs deployment precision (still unmeasured).</li>
<li><strong>Split-leakage fixed</strong> — the RegionName hotfix + re-split landed (leakage-free, 0 ecoregions
span splits). Phase-0/3/5 numbers stay on the leaky split (relative-only); Phase-4 + the final test use the
corrected split.</li>
<li><strong>Feasibility</strong> — pan-arctic mapping is realistic as a <strong>QC-assisted candidate map</strong>
(high recall + filtering), not yet a fully-automated high-precision product.</li>
</ul>
"""


def _section_future() -> str:
    return """
<h2 id='future'>9. Open questions, future work &amp; decisions</h2>
<ul>
<li><strong>Spatial generalization</strong> — leave-one-ecoregion-out CV (make-or-break for pan-arctic).</li>
<li><strong>Honest deployment precision</strong> — false-positives per true-positive on a held-out region at realistic prevalence.</li>
<li><strong>EXTRA channels (Phase 4)</strong> — the primary plateau-breaker; S2 bands generated, SE path next.</li>
<li><strong>RegionName/split hotfix</strong> — re-split on corrected metadata; re-score μ₀ if the val split moves.</li>
<li><strong>Inference</strong> — us-west1 16× L4 fleet decided (PR #19 merged); pan-arctic pass after winner lock.</li>
<li><strong>Multi-year consistency</strong> (2024∧2025) as a precision lever at deployment.</li>
</ul>
"""


def _section_family_learnings() -> str:
    """Per-experiment-group ('family') 'what we learned' + the cross-cutting thesis.

    Curated from the ledger SSoT (docs/experiment_ledger.md). This is the narrative the
    detailed per-phase tables below support."""
    cross = "".join(
        f"<div class='insight'><strong>{t}</strong><br>{body}</div>"
        for t, body in CROSS_CUTTING
    )
    rows = ""
    for f in FAMILY_LEARNINGS:
        rows += f"""
<tr>
  <td style='text-align:center; font-weight:700; font-size:1.05rem; color:#1d4ed8'>{f['id']}</td>
  <td><strong>{f['name']}</strong><br>{_badge(f['status'])}</td>
  <td>{f['learned']}</td>
  <td style='font-size:0.82rem; color:#475569'>{f['evidence']}</td>
</tr>"""
    return f"""
<h2 id='learnings'>What each experiment group taught us</h2>
<p>The project is organised into experiment <strong>families A–K</strong> (the SSoT registry is
<code>docs/experiment_ledger.md</code>). This section is the headline takeaway from each group; the
detailed run tables and curves are in the per-phase sections below.</p>
<h3>Cross-cutting thesis</h3>
{cross}
<h3>By family</h3>
<table>
  <thead><tr><th style='width:3%'>Fam</th><th style='width:17%'>Group</th>
  <th style='width:42%'>What we learned</th><th style='width:38%'>Deciding evidence</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _section_locked_decisions() -> str:
    """Every locked choice + the family it belongs to + how it was decided."""
    rows = ""
    for name, choice, fam, how in LOCKED_DECISIONS:
        rows += (f"<tr><td><strong>{name}</strong></td>"
                 f"<td class='winner'>{choice}</td>"
                 f"<td style='text-align:center'>{fam}</td>"
                 f"<td style='font-size:0.85rem; color:#475569'>{how}</td></tr>\n")
    return f"""
<h2 id='locked'>Locked decisions &amp; how each was decided</h2>
<p>The v2 recipe is the accumulation of these locked choices. Each was gated on
Δ ≥ G=({GATE_G:.4f}) with a precision-@-recall guard; every LOCK additionally required 3-seed
sign-consistency (corrected-split σ≈{SIGMA_CORRECTED}).</p>
<table>
  <thead><tr><th style='width:20%'>Decision</th><th style='width:28%'>Locked choice</th>
  <th style='width:5%'>Fam</th><th>How it was decided</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<div class='card'><span class='label'>v2 deploy reference (corrected split)</span><br>
<span class='metric'>{DEPLOY_MEAN:.4f}</span>
<span style='font-size:0.85rem'>3-seed mean — {DEPLOY_SEEDS[0]} / {DEPLOY_SEEDS[1]} / {DEPLOY_SEEDS[2]}</span></div>
"""


def _count_runs() -> dict:
    """Dynamic dashboard counts from the run dirs (host or container path)."""
    out = {"run_dirs": 0, "summaries": 0}
    for base in ("/mnt/outputs/v1.0/runs", "/outputs/v1.0/runs"):
        p = Path(base)
        if p.is_dir():
            out["run_dirs"] = sum(1 for d in p.iterdir() if d.is_dir())
            out["summaries"] = sum(1 for _ in p.glob("*/run_summary.md"))
            break
    return out


def _section_training_overview() -> str:
    """Program dashboard: how much was explored, what's locked vs still open."""
    c = _count_runs()
    n = c["run_dirs"] or 96  # ledger master table ≈ this many launched runs
    cards = [
        ("Training runs launched", str(n), "across families A–K (ledger master table)"),
        ("Architectures tried", "8+", "UNet++/EffB5 (locked), FPN, DeepLabV3+, PSPNet, MANet, EffB3, "
         "DINOv3 (web+sat), SAM2/Hiera"),
        ("Loss × boundary cells", "10", "focal/compound/tversky × ignore-width {1,2,3}"),
        ("EXTRA-channel configs", "12+", "RGB, NDVI, NBR, TC, SE-PCA, SE-Proto, full-8band + greedy pairs"),
        ("Augmentation arms", "12+", "photometric audit, RandomScale A/B, copy-paste/mosaic/cutmix/mixup, "
         "RandAug/TrivialAug"),
        ("Fusion methods", "5", "F0–F5 (F0 locked; F3/F5 heavy fusion tested → lose)"),
    ]
    card_html = "".join(
        f"<div style='flex:1 1 200px'><span class='label'>{lab}</span><br>"
        f"<span class='metric'>{val}</span><br>"
        f"<span style='font-size:0.8rem; color:#64748b'>{sub}</span></div>"
        for lab, val, sub in cards
    )
    locked = sum(1 for f in FAMILY_LEARNINGS if f["status"] == "done")
    running = sum(1 for f in FAMILY_LEARNINGS if f["status"] == "running")
    pending = sum(1 for f in FAMILY_LEARNINGS if f["status"] == "pending")
    return f"""
<h2 id='dashboard'>Training overview</h2>
<div class='card'><div style='display:flex; gap:1.5rem; flex-wrap:wrap'>{card_html}</div></div>
<p style='font-size:0.9rem'>Families: <strong>{locked} settled</strong> ·
<strong>{running} in-progress</strong> · <strong>{pending} pending/conditional</strong>.
Hyperparameters that are <strong>locked</strong> (loss, boundary, channels, fusion, backbone, sampling,
aug) vs <strong>yet-to-lock</strong> (calibration/TTA, final encoder verdict) are detailed in the
<a href='#locked'>Locked decisions</a> section. Split caveat: phases 0/2/3/5 are on the leaky split
(relative only); families 4/10/D/E/F/I and the final test use the corrected leakage-free split.</p>
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the project living HTML report from MLflow")
    p.add_argument("--config", default="configs/baseline.yaml",
                   help="Config YAML for MLflow experiment_name")
    p.add_argument("--tracking-uri", default=None,
                   help="Override mlflow.tracking_uri from config (e.g. /mnt/outputs/mlflow)")
    p.add_argument("--output", default="docs/report.html",
                   help="Output HTML file path")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    cfg = load_config(args.config)
    tracking_uri = args.tracking_uri or cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

    logger.info("Connecting to MLflow: %s / %s", tracking_uri, experiment_name)
    mlflow = _connect_mlflow(tracking_uri)

    import datetime
    overview = _section_overview(tracking_uri)
    learnings = _section_family_learnings()
    locked = _section_locked_decisions()
    dashboard = _section_training_overview()
    s0a = _section_phase0a(mlflow, experiment_name)
    s0b = _section_phase0b(mlflow, experiment_name)
    s0c = _section_phase0c(mlflow, experiment_name)
    s1 = _section_phase1()
    s2 = _section_phase2(mlflow, experiment_name)
    s3 = _section_phase3(mlflow, experiment_name)
    s45 = _section_phase45(mlflow, experiment_name)
    findings = _section_findings()
    future = _section_future()
    s_art = _section_artifacts(mlflow, experiment_name)
    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RTS Segmentation v2 — Project Report</title>
<style>{_CSS}</style>
</head>
<body>
<h1>RTS Segmentation v2 — Project Report</h1>
<p style="color:#64748b">Living dashboard for the pan-arctic RTS mapping project. Auto-generated
{now} from MLflow <code>{tracking_uri}</code> (experiment <code>{experiment_name}</code>). Rules:
<code>docs/report.md</code> · program SSoT: <code>training/experiments.md</code>.</p>

{overview}
{learnings}
{locked}
{dashboard}

<hr style="margin-top:2.5rem">
<p style="color:#64748b; font-size:0.9rem"><strong>Detailed run history</strong> — the per-phase tables,
curves and figures the narrative above is built from (auto-generated from MLflow; phases use the original
numbering, mapped to families A–K in the ledger).</p>
<h2 id='p0a'>2. Phase 0 — Baseline calibration <span class='badge b-done'>done</span></h2>
{s0a}
{s0b}
{s0c}
{s1}
{s2}
{s3}
{s45}
{findings}
{future}

<hr style="margin-top:3rem">
{s_art}
<p style="color:#94a3b8; font-size:0.8rem">
Generated by <code>scripts/build_report.py</code>. Regenerate after each run / phase per
<code>docs/report.md</code>.
</p>
</body>
</html>"""

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    logger.info("Report written to %s", out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
