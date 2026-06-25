# Experiment Ledger — RTSmapping_v2

**This file is the experiments SSoT.** Every training run, the locked recipe, the findings, and the
dropped ideas live here and nowhere else. `docs/report.html` is the generated analytical/visual view of
this file; `current_working_status.md` is the project diary. To update, follow the ritual in `CLAUDE.md`.

**Metric:** `val_realistic_pr_auc_geomean` = `best_smoothed` (higher is better). The **`score` column of
the run table is machine-harvested** from each run's `run_summary.json` by `scripts/sync_experiments.py`
— do not hand-edit it. Everything else is agent-edited.

**Split warning:** scores are **not comparable across the split boundary.** Families A/B/C-loss/E-decoder
ran on the **leaky** region split (relative comparisons only). D/F/G/I and the encoder runs use the
**corrected** leakage-free split (absolute numbers sit higher; compare within-family vs the in-family
control, not across phases). Test-Realistic is scored once, honestly, on the corrected split.

**Status:** done · seed-confirm · running · incomplete (killed before maturity) · crashed · collapsed ·
degenerate · killed (ran to a verdict but failed) · lr-test.

**Families:** A Baseline/gate · B Data · C Loss/boundary · D Channels/fusion · E Architecture/encoder ·
F Augmentation · G Sampling · H Calibration/TTA · I Final-lock/Test · J Deploy/inference · K Deferred.

<!-- GATE:BEGIN -->
## Gate

| μ₀ | σ₀ (leaky) | **G = max(2σ₀, 0.01)** | σ (corrected) |
|----|-----------|------------------------|---------------|
| 0.7912 | 0.0056 | **0.0112** | ≈0.012 |

**Lock policy:** a change is locked only if a 3-seed confirm shows **mean Δ ≥ G** *and*
**sign-consistency across all 3 seeds** (G alone is a single-seed *screen*). Sign-consistency is the
decisive test — it separated drop-RandomScale (+0.016, 3/3 → locked) from curriculum r20_pf33 (+0.006,
sign-flipped → rejected).
<!-- GATE:END -->

---

<!-- RUN-TABLE:BEGIN — `score` is harvested by scripts/sync_experiments.py; do not hand-edit that column. One run-dir name per row. -->
## Master run table

| name | fam | split | score | status | note |
|------|:---:|:-----:|------:|:------:|------|
| phase0a_arm_a | A | leaky | 0.6666 | done | norm arm A (per-dataset z-score) — LOCKED |
| phase0a_arm_b | A | leaky | 0.6261 | done | norm arm B (x255+ImageNet) |
| phase0a_arm_c | A | leaky | 0.6697 | done | norm arm C (x255-only) |
| phase0b_lr_frozen | A | leaky | 0.0008 | degenerate | LR probe (frozen) |
| phase0b_lr_unfrozen | A | leaky | — | crashed | LR probe (unfrozen) — diverged |
| phase0c_seed42 | A | leaky | 0.7899 | done | 3-seed baseline → μ₀=0.7912, σ₀=0.0056, G=0.0112 |
| phase0c_seed43 | A | leaky | 0.7863 | done | 3-seed baseline |
| phase0c_seed44 | A | leaky | 0.7973 | done | 3-seed baseline |
| phase2_scale_25 | B | leaky | 0.7636 | done | data-scale 25% |
| phase2_scale_50 | B | leaky | 0.7720 | done | data-scale 50% |
| phase2_scale_75 | B | leaky | 0.7916 | done | data-scale 75% → plateau (slope flat) |
| phase3_loss_compound_1to1 | C | leaky | 0.7878 | done | loss sweep |
| phase3_loss_compound_2to1 | C | leaky | 0.7933 | done | loss sweep |
| phase3_loss_compound_1to2 | C | leaky | 0.7998 | done | near-miss (Δ<G) → carried to boundary factorial |
| phase3_loss_compound_1to2_seed43 | C | leaky | 0.7872 | done | seed-confirm |
| phase3_loss_tversky_3to7 | C | leaky | 0.5902 | done | Tversky 3:7 |
| phase3_loss_tversky_2to8 | C | leaky | 0.0729 | collapsed | Tversky 2:8 collapse |
| phase3_bd_focal_ignore_w1 | C | leaky | 0.7872 | done | boundary factorial |
| phase3_bd_focal_ignore_w2 | C | leaky | 0.8046 | done | **boundary winner → LOCKED (focal·ignore_w2)** |
| phase3_bd_focal_ignore_w2_seed43 | C | leaky | 0.8200 | done | seed-confirm |
| phase3_bd_focal_ignore_w2_seed44 | C | leaky | 0.8054 | done | seed-confirm |
| phase3_bd_focal_ignore_w3 | C | leaky | 0.7973 | done | boundary factorial |
| phase3_bd_compound_1to2_ignore_w1 | C | leaky | 0.7996 | done | boundary factorial |
| phase3_bd_compound_1to2_ignore_w2 | C | leaky | 0.8003 | done | boundary factorial |
| phase3_bd_compound_1to2_ignore_w3 | C | leaky | 0.8025 | done | close runner-up |
| phase3_bd_compound_1to2_ignore_w3_seed43 | C | leaky | 0.8170 | done | seed-confirm |
| phase3_bd_compound_1to2_ignore_w3_seed44 | C | leaky | 0.8153 | done | seed-confirm |
| ablation_noignore_ndvi_seed42 | C | corrected | 0.8727 | done | ignore-region ablation (train-only) — ignore helps |
| ablation_noignore_ndvi_seed43 | C | corrected | 0.9214 | done | seed-confirm |
| ablation_noignore_ndvi_seed44 | C | corrected | 0.9012 | done | seed-confirm (mean 0.8984, Δ−0.014 vs deploy) |
| phase4_extra_rgb_baseline | D | corrected | 0.8297 | done | **RGB control (corrected-split anchor 0.830)** |
| phase4_extra_ndvi | D | corrected | 0.8879 | done | **best single channel; 3-seed mean 0.8985** |
| phase4_extra_ndvi_seed43 | D | corrected | 0.8965 | done | seed-confirm |
| phase4_extra_ndvi_seed44 | D | corrected | 0.9111 | done | seed-confirm |
| phase4_extra_ndvi_fastcheck | D | corrected | 0.8934 | done | base_v2_fast stop-fix validation (gate-neutral) |
| phase4_extra_full | D | corrected | 0.8763 | done | full 8-band ≈ NDVI-alone |
| phase4_extra_full_seed43 | D | corrected | 0.8619 | done | seed-confirm |
| phase4_extra_full_seed44 | D | corrected | 0.8678 | done | seed-confirm |
| phase4_extra_nbr | D | corrected | 0.8469 | done | single-group NBR |
| phase4_extra_tc | D | corrected | 0.8683 | done | single-group tasseled-cap |
| phase4_extra_se_pca | D | corrected | 0.8736 | done | single-group SE-PCA |
| phase4_extra_se_pca_seed43 | D | corrected | 0.8571 | done | seed-confirm |
| phase4_extra_se_proto | D | corrected | 0.8468 | done | single-group SE-prototype |
| phase4_extra_ndvi_nbr | D | corrected | 0.8559 | done | greedy round-1 NDVI+NBR (no add) |
| phase4_extra_ndvi_tc | D | corrected | 0.8996 | done | greedy round-1 NDVI+TC (no add) |
| phase4_extra_ndvi_sepca | D | corrected | 0.8963 | done | NDVI+SE-PCA |
| phase4_extra_ndvi_sepca_seed43 | D | corrected | 0.8949 | done | seed-confirm |
| phase4_extra_ndvi_sepca_seed44 | D | corrected | 0.9088 | done | seed-confirm (mean 0.900 ≈ NDVI-alone) |
| phase4_extra_ndvi_seproto | D | corrected | 0.8984 | done | NDVI+SE-proto |
| phase4_extra_ndvi_seproto_seed43 | D | corrected | 0.8988 | done | seed-confirm |
| phase4_extra_ndvi_seproto_seed44 | D | corrected | 0.8966 | done | seed-confirm (mean 0.898 ≈ NDVI-alone) |
| phase4_f1_full | D | corrected | 0.8903 | done | F1 smart-stem-init (8-band) ≈ NDVI-alone |
| phase4_f1_ndvi_seproto | D | corrected | 0.8921 | done | F1 (pair) ≈ NDVI-alone |
| phase4_f2_full | D | corrected | 0.8268 | done | F2 channel-attn collapses on 8-band |
| phase4_f2_ndvi_seproto | D | corrected | 0.8891 | done | F2 (pair) < NDVI-alone |
| phase4_f3_full | D | corrected | 0.8184 | done | F3 dual-encoder late fusion — loses to F0 |
| phase4_f5_full | D | corrected | 0.8480 | done | F5 residual cross-modal attn (JSTARS) — loses to F0 |
| phase4_f5_ndvi_seproto | D | corrected | 0.8544 | done | F5 (pair) — loses to F0 |
| phase5_arch_fpn | E | leaky | 0.7939 | done | decoder sweep — ties UNet++ |
| phase5_arch_deeplabv3plus | E | leaky | 0.7878 | done | decoder sweep |
| phase5_arch_pspnet | E | leaky | 0.7288 | done | decoder sweep |
| phase5_arch_manet | E | leaky | 0.6213 | done | decoder sweep (worst) |
| effb3_deploy | E | corrected | 0.9050 | done | capacity-down probe — no-win (cheaper fallback) |
| phase4_fm_dinov3_rgb | E | corrected | 0.8734 | done | web-DINOv3 ViT-B RGB (z-score norm) |
| phase4_fm_dinov3_rgb_lrtest | E | corrected | — | lr-test | LR range test |
| fm_dinov3_rgb_imagenet | E | corrected | 0.8923 | done | web-DINOv3 RGB + ImageNet norm (de-confound) |
| phase4_fm_dinov3_ndvi | E | corrected | 0.9121 | done | web-DINOv3+NDVI **ties EffB5** → generic FM not the lever |
| fm_sam2_rgb | E | corrected | 0.5558 | done | SAM2/Hiera — non-competitive |
| fm_dinov3sat_7b_frozen | E | corrected | 0.4747 | killed | sat-7B frozen — diverged, non-competitive |
| fm_dinov3sat_l_rgb | E | corrected | 0.9200 | done | sat-DINOv3 ViT-L RGB, 3-seed |
| fm_dinov3sat_l_rgb_seed43 | E | corrected | 0.9195 | done | seed-confirm |
| fm_dinov3sat_l_rgb_seed44 | E | corrected | 0.9003 | done | seed-confirm (mean 0.9133 ≈ EffB5) |
| fm_dinov3sat_l_ndvi | E | corrected | 0.9234 | done | sat-DINOv3 ViT-L +NDVI, seed42 (**off-recipe — confounded**) |
| fm_dinov3sat_l_ndvi_seed43 | E | corrected | 0.9199 | done | seed-confirm |
| fm_dinov3sat_l_ndvi_seed44 | E | corrected | 0.9150 | done | seed-confirm (mean 0.9194, Δ+0.0071 sub-gate) |
| fm_dinov3sat_l_ndvi_seed42_rerun | E | corrected | — | crashed | off-recipe rerun (superseded by _locked) |
| fm_dinov3sat_l_ndvi_locked | E | corrected | 0.9221 | done | **FAIR sat re-run (locked recipe), seed42 — peak ep40** |
| fm_dinov3sat_l_ndvi_locked_seed43 | E | corrected | 0.9286 | done | seed-confirm — peak ep40 |
| fm_dinov3sat_l_ndvi_locked_seed44 | E | corrected | 0.9067 | done | seed-confirm — peak ep35 (**mean 0.9191 ≈ EffB5 0.9218 → TIE → deploy EffB5**) |
| aug_ref | F | corrected | 0.8661 | done | aug control |
| aug_ref_seed43 | F | corrected | 0.8468 | done | seed-confirm |
| aug_ref_seed44 | F | corrected | 0.8808 | done | seed-confirm (ref mean 0.865) |
| aug_p0_geom_only | F | corrected | 0.7936 | done | geometric-only → photometric matters (−0.072) |
| aug_p1_no_clahe | F | corrected | 0.8541 | done | drop CLAHE (−0.012, within noise) |
| aug_p3_photo_x15 | F | corrected | 0.8658 | done | photometric ×1.5 ≈ ref (no gain) |
| aug_pad_ignore | F | corrected | 0.8527 | done | pad-ignore fix — downscale itself hurts, not the pad |
| aug_scale_off | F | corrected | 0.8862 | done | **drop RandomScale → LOCKED (+0.016, 3/3)** |
| aug_scale_off_seed43 | F | corrected | 0.8673 | done | seed-confirm |
| aug_scale_off_seed44 | F | corrected | 0.8892 | done | seed-confirm (mean 0.881) |
| aug_copypaste_deploy | F | corrected | 0.8930 | done | mixing aug — worst (breaks shadow/context cues) |
| aug_mosaic_deploy | F | corrected | 0.9069 | done | mixing aug — no-win |
| aug_cutmix_deploy | F | corrected | 0.9014 | done | mixing aug — no-win |
| aug_mixup_deploy | F | corrected | 0.9028 | done | mixing aug — no-win (family 4/4 struck out) |
| aug_randaugment_deploy | F | corrected | 0.9089 | done | shadow-safe pool — no-win (−0.0034) |
| aug_trivialaugment_deploy | F | corrected | 0.9167 | done | **TrivialAugment → LOCKED color stage (3-seed mean 0.9218)** |
| aug_trivialaugment_deploy_seed43 | F | corrected | 0.9216 | done | seed-confirm |
| aug_trivialaugment_deploy_seed44 | F | corrected | 0.9270 | done | seed-confirm |
| aug_anneal_deploy | F | corrected | 0.9074 | done | aug-strength annealing — no-win |
| aug_anneal_deploy_seed43 | F | corrected | 0.9192 | done | seed-confirm |
| aug_anneal_deploy_seed44 | F | corrected | 0.9204 | done | seed-confirm (mean 0.9157, Δ+0.0034 sub-gate) |
| phase10_curric_base | G | corrected | 0.8786 | done | curriculum control |
| phase10_curric_r20_pf33 | G | corrected | 0.8945 | done | best cell — but seed-confirm = noise → rejected |
| phase10_curric_r20_pf33_seed43 | G | corrected | 0.9013 | done | seed-confirm |
| phase10_curric_r20_pf33_seed44 | G | corrected | 0.8587 | done | seed-confirm (sign-flipped → noise) |
| phase10_curric_r30_pf33 | G | corrected | 0.8528 | done | curriculum cell |
| phase10_curric_r30_pf50 | G | corrected | 0.8530 | done | curriculum cell |
| phase_lock_ndvi_bd_curric | I | corrected | 0.9063 | done | early lock attempt — superseded by deploy_v1 |
| deploy_v1_ndvi_seed42 | I | corrected | 0.9144 | done | **final-lock 3-seed (v2 recipe) — reference baseline** |
| deploy_v1_ndvi_seed43 | I | corrected | 0.9068 | done | final-lock 3-seed |
| deploy_v1_ndvi_seed44 | I | corrected | 0.9156 | done | final-lock 3-seed (mean 0.9123) |<!-- RUN-TABLE:END -->

---

<!-- RECIPE-TABLE:BEGIN -->
## v2 final recipe (locked)

| Component | Locked choice | Fam | Evidence |
|---|---|:---:|---|
| Channels | RGB + **NDVI** (4-ch), F0 early channel-stack | D | NDVI 0.8985 ≫ RGB 0.830; greedy adds nothing; F1/F2 tie, F3/F5 lose |
| Normalization | per-dataset z-score (Arm A) | A | Arm A > B/C |
| Decoder | **UNet++** (dense skips) | E | UNet++ ≥ FPN > DeepLabV3+ > PSPNet ≫ MAnet — none beats it |
| Encoder | **EffB5** (UNet++/EfficientNet-B5) | E | fair re-run: sat-DINOv3 ViT-L **ties** (0.9191 vs 0.9218, obj metrics equal) → no benefit, EffB5 ~4× cheaper |
| Loss + boundary | focal + **ignore_w2** | C | boundary factorial, seed-confirmed |
| Sampling | default balanced (no curriculum) | G | curriculum rejected (sign-flipped) |
| Augmentation | geometric + **TrivialAugment** − **RandomScale** | F | drop-scale +0.016 3/3; TrivialAugment 0.9218 |

**Training schedule** (reproduce-training only, not a deploy param): `base_v2_fast` — patience 5,
start_epoch 45, max_epochs 120; bf16; seeds 42/43/44; deterministic.

**Deploy/inference** (`configs/deployment.yaml`): threshold + temperature + TTA ⏳ *(set by H
calibration, still null)*; stride 344 px (~33% overlap); overlap fusion = distance-from-center Gaussian
σ=128 px; NDVI windowed on-the-fly from S2 composites (inference.md §3.3/§4.3).
<!-- RECIPE-TABLE:END -->

<!-- BUILDUP-TABLE:BEGIN -->
## Recipe build-up (PR-AUC, `best_smoothed`, 3-seed)

| Step (cumulative) | PR-AUC | Δ step | Δ vs baseline |
|---|---:|---:|---:|
| RGB baseline | 0.830 | — | — |
| + NDVI | 0.8985 | +0.069 | +0.069 |
| + boundary ignore_w2 + drop-RandomScale (= deploy_v1) | 0.9123 | +0.014 | +0.082 |
| + TrivialAugment (current EffB5 recipe) | 0.9218 | +0.0095 | +0.092 |
| sat-DINOv3 encoder (fair re-run, same recipe) | 0.9191 | −0.003 (tie) | — → **EffB5 deployed** (sat no benefit, 4× costlier) |
<!-- BUILDUP-TABLE:END -->

---

<!-- FINDINGS:BEGIN -->
## Findings (per family)

**A — Baseline & gate.** A reproducible 3-seed baseline (μ₀=0.7912, σ₀=0.0056) sets the winner gate
G=0.0112. Re-measured on the corrected split, σ≈0.012 (~2×) → every lock needs mean Δ≥G **and** 3-seed
sign-consistency. Normalization Arm A (per-dataset z-score) locked.

**B — Data scaling.** More labeled data is **not** the near-term lever: v1.0 is on a data plateau
(25→100%: 0.764→0.792, flat slope) and the model is well-matched to its data (train/val IoU gap
0.05/0.17 ≪ 0.4). No new labels are coming → squeeze the fixed ~1.5k positives via representation, not
volume. This is the central diagnosis: **representation-limited, not data-volume- or capacity-limited.**

**C — Loss & boundary.** The win is the **boundary treatment, not the loss.** Focal alone is the loss
winner; precision-skewed Tversky collapses (2:8 → 0.073). Adding a boundary-ignore band clears the gate
— **focal·ignore_w2 0.8046** (seed-confirmed 0.805–0.820), tied with compound 1:2·ignore_w3
(0.802–0.817); focal·w2 chosen for simplicity. The ignore-region ablation (train-only) confirms ignore
helps (mean 0.8984, Δ−0.014 vs deploy; caveat: positives overwrite ignore, so not a clean counterfactual).

**D — Channels & fusion.** A single well-chosen channel (**NDVI**) is the biggest representation win,
and more is not better. NDVI-alone 3-seed **0.8985 ≫ RGB 0.830** (+0.07, ≫σ) and > full 8-band (0.869).
Greedy forward from NDVI adds nothing (+SE-PCA 0.900, +TC 0.900, +SE-proto 0.898, +NBR 0.856 — none
clears G). Heavy fusion **loses**: F3-full 0.818, F5-full 0.848, F5-pair 0.854 (all ≪ NDVI-alone; below
even the F0 stack). **LOCKED: EXTRA=[NDVI], F0 early channel-stack.**

**E — Architecture & encoder (RESOLVED 2026-06-25 → EffB5).** Capacity is not the lever, and neither is
the encoder. No CNN decoder beats UNet++ (FPN 0.794 ties > DeepLabV3+ 0.788 > PSPNet 0.729 ≫ MAnet 0.621);
EffB3 capacity-down no-win (0.9050). Generic web-DINOv3+NDVI ties EffB5 (0.9121 vs 0.9123). The
satellite-pretrained DINOv3 ViT-L *looked* like a breakout off-recipe (+NDVI 0.9194 + big object metrics)
— **but that was the confound**: sat ran on `phase0c` (boundary-none val labels, no TrivialAugment) and was
compared to the pre-TrivialAugment EffB5 0.9123. The **fair re-run on the full locked recipe** (ignore_w2 +
drop-RandomScale + TrivialAugment, *identical* val labels) settles it — **a dead tie**:

| 3-seed, **locked recipe** (identical val labels) | PR-AUC | pixel-IoU | obj-F1 |
|---|---:|---:|---:|
| EffB5 (= `aug_trivialaugment_deploy`) | **0.9218** | 0.612 | 0.438 |
| sat-DINOv3 ViT-L (= `fm_dinov3sat_l_ndvi_locked`) | 0.9191 | 0.612 | 0.437 |
| Δ (sat − EffB5) | −0.0027 | ≈0 | ≈0 |

> **Verdict: DEPLOY EffB5.** On the matched recipe the satellite ViT-L gives **no benefit** on any metric,
> and EffB5 (CNN) is ~4× cheaper/faster across the 41.57M-tile pass. The off-recipe "sat edge"
> (+0.13 IoU / +0.07 obj-F1 on boundary-none labels) **collapses** once both use the same recipe + val
> labels — a textbook confound, caught by the fair A/B. SAM2 (0.556) and sat-7B-frozen (0.475)
> non-competitive. The sat re-run (`fm_dinov3sat_l_ndvi_locked*`) ran to ep60, peaked ep35–40
> (best_smoothed 0.9221/0.9286/0.9067), terminated in the overfit tail — a complete verdict.

**F — Augmentation.** Not a plateau-breaker, but two cheap wins lock in: **drop RandomScale** (+0.016,
3/3 seeds) and replace the hand-tuned color stack with **TrivialAugment** (3-seed mean 0.9218, 3/3 >
deploy; Δ+0.0095 just under G but locked by judgment — a parameter-free auto-policy that's lighter and
consistent). Photometric aug matters (geometric-only craters to 0.794, −0.072). All other arms fail:
mixing family 4/4 (copy-paste 0.893 worst, cutmix 0.901, mixup 0.903, mosaic 0.907), RandAugment 0.909,
aug-anneal 0.916.

**G — Sampling.** Default balanced sampling is sufficient — the curriculum r20_pf33 "win" (0.894
single-seed) did not survive seeds (0.894/0.901/0.859, sign-flipped) → rejected as noise.

**H — Calibration & TTA (DONE 2026-06-25, on Val-Realistic).** `scripts/calibrate.py` on the 3 EffB5 seeds:
**TTA → none** (D4-TTA *hurt* 0.9302→0.9234; hflip +0.0014 < 1% gate). **Temperature T≈0.51–0.54** (<1 — the
focal-trained model is *under*-confident, so calibration sharpens logits; threshold lands low ~0.12–0.16).
Per-seed PR-AUC-geomean 0.9161 / 0.9216 / 0.9302 (P≈0.80/R≈0.86–0.88). **3-seed mean-prob ensemble = 0.9393**
(P=0.800/R=0.896). **Deploy = the 3-seed ensemble** (T=0.5123, tta=none → `configs/deployment.yaml`):
chosen for robustness against an unlucky single seed (0.916 vs 0.930), not the marginal +0.0091 (which is
sub-gate).

**H.2 — Object operating point (DONE 2026-06-25, `scripts/tune_object_operating_point.py`, Val-Realistic).**
The calibrate.py threshold (0.1224) is tuned for *pixel* precision and is the **wrong operating point for an
object product**: at thr 0.1224 / min_blob 10 the ensemble scores obj-F1 **0.304** (obj-P 0.189, 443 FP objects,
424 of them no-overlap speckle). Sweeping threshold × min_blob × morph-close picks the **obj-F1 argmax at the
pixel-P≥0.8 floor: thr 0.30 + min_blob 80 + morph off** → obj-F1 **0.567** (obj-P 0.489 / obj-R 0.674, pixel-P
0.931, FP objects 443→93). Robust plateau (obj-F1≈0.56 over thr 0.30–0.35; morph radius 0/1/2 identical →
off). Precision-leaning alternative: thr 0.65 → obj-P 0.61 / obj-R 0.44 (obj-F1 0.511). **Adopted thr 0.30 /
min_blob 80** into `deployment.yaml` (precision-over-recall, balanced obj-F1). Report-only tool; the operating
point is frozen on Val and reversible before the one-shot. Test-Realistic (held) gives the honest number.

**I — Final lock.** **Encoder = EffB5** (fair sat-DINOv3 re-run tied, 0.9191 vs 0.9218, equal object metrics →
no benefit at ~4× cost). v2 recipe (RGB+NDVI · F0 · focal·ignore_w2 · default sampling · aug−RandomScale ·
**TrivialAugment** · base_v2_fast) 3-seed **0.9218** (= `aug_trivialaugment_deploy`, 3 clean checkpoints),
deployed as a **3-seed ensemble** (calibration in H). Remaining: build the ensemble deploy/eval path
(per-seed packages + fusion manifest; `predictor.py`/`evaluate_test.py` multi-model) → the one-shot
Test-Realistic → package. No foundation adapter needed.
<!-- FINDINGS:END -->

---

## Dropped & discussed-but-didn't-land

| Idea | Fam | Verdict / why |
|---|:---:|---|
| Curriculum r20_pf33 | G | tested → rejected (within seed noise, sign-flipped) |
| RandomScale downscale aug | F | tested → dropped (removing it +0.016, all seeds) |
| Mixing augs (copy-paste / mosaic / cutmix / mixup) | F | tested → no-win (4/4; copy-paste worst) |
| RandAugment · aug-strength annealing | F | tested → no-win (sub-gate) |
| F2 channel-attention (8-band) | D | tested → collapsed (0.827) |
| F3 dual-encoder / F5 cross-modal attn | D | tested → lose to F0 (heavy fusion extracts less than the stack) |
| EffB3 capacity-down | E | tested → no-win (0.9050); kept as a cheaper deploy fallback only |
| SAM2 / sat-7B-frozen | E | tested → non-competitive (0.556 / 0.475) |
| Web-DINOv3 + EXTRA | E | tested → ties EffB5 once NDVI is in → generic FM not the lever |
| §6.5 loss×wd×curriculum interaction | C/F/G | dropped (moot after wd dropped + curriculum rejected + loss locked) |
| wd × aug regularization grid | F | dropped (over-parameterization trigger never fired) |
| SegFormer (mit_b5) · EffB7 · UNet3+ | E | dropped (low value / overfit risk on a plateau / condition unmet) |
| YOLO / Mask R-CNN | E | rejected (paradigm mismatch) |
| SAM3 | E | dropped (image incompatible — py3.12/torch2.7) |
| Re-run Phase 2 on 3500 positives | B | moot (no new labels) |
| Pseudo-labeling / self-training | K | backup only (confirmation-bias risk) |
| Soft-label boundary handling | C | deferred (ignore covers annotation noise for v2) |

**Deferred to v3 (K):** v1.0 re-stage (+28 pos / −49 black) · hard-negative mining (post first
inference) · MAE SSL pretraining (user-go, end-stage). **Conditional:** scale-TTA · ensemble (decided
at final lock) · context-expansion multi-scale (post-inference map review).
