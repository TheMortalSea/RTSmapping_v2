# Experiment Ledger

Master chronological registry of **every** training run in RTSmapping_v2, with status and score.
Living doc — update when runs finish, launch, or change state. SSoT for "what has been tried."

**Metric:** `val_realistic_pr_auc_geomean` (= `best_smoothed` in each `run_summary.md`). Higher is better.
Source of truth: `/mnt/outputs/v1.0/runs/<name>/run_summary.md` (finished) and `.../logs/<name>.log` (live).

**Status legend:** ✅ done · 🔵 ongoing (live best so far) · ⏳ queued · ⏸️ deferred · ❌ dropped/crashed.

> ⚠️ **Scores are NOT comparable across the split boundary.** Phases 0/2/3/5 ran on the **leaky**
> region split (relative comparisons only, by design). Phases 4/10 run on the **corrected leakage-free**
> split — their absolute numbers sit higher and must be compared *within phase* against the in-phase
> control (Phase-4 RGB control = 0.830), not against earlier phases. Final test is scored once, honestly,
> on the corrected split (Step 5).

_Last refreshed: 2026-06-24 — **v2 campaign CLOSED (run-complete); repo consolidated to only-`main`.** Encoder verdict (3-seed, peak-epoch val): **sat-DINOv3 ViT-L + NDVI ~0.9224 PR-AUC** (0.9274/0.9199/0.9198) vs EffB5+NDVI **0.9123** — +0.0101 (near-miss on G=0.0112) but **decisive on object metrics: pixel-IoU 0.64 vs 0.51, obj-F1 0.56 vs 0.39**. sat-DINOv3-**RGB** ties +NDVI on PR-AUC (~0.913) with 3 clean ckpts ready; **+NDVI** marginally better (+0.008 PR-AUC, +0.02 obj-F1) — user opted to **include NDVI** and decide solo-vs-ensemble on Val at Phase D. seed42 +NDVI ckpt was lost to the resume-clobber bug (now fixed) → **retraining**. Prior screens all closed: greedy EXTRA=NDVI; F3/F5 heavy fusion loses (F0 locked); mixing-aug 4/4 + EffB3 no-win; web-DINOv3+NDVI 0.9120 ties EffB5 (generic foundation not the lever); SAM2/Hiera 0.590 + 7B-frozen 0.498 non-competitive; ignore-region ablation PAUSED (needs RTS-truth source). (Model = v2; repo RTSmapping_v2.)_

---

## Master table (chronological by launch)

| # | Started | Experiment | Phase / purpose | Split | Score | Status |
|---|---------|-----------|-----------------|-------|------:|:------:|
| 1 | 06-13 08:00 | phase0b_lr_frozen | 0b LR probe (frozen) | leaky | 0.0008 | ❌ degenerate |
| 2 | 06-13 11:12 | phase0b_lr_unfrozen | 0b LR probe (unfrozen) | leaky | — | ❌ crashed |
| 3 | 06-13 23:37 | phase0a_arm_a | 0a norm arm A (z-score) | leaky | 0.667 | ✅ |
| 4 | 06-14 01:46 | phase0a_arm_b | 0a norm arm B | leaky | 0.626 | ✅ |
| 5 | 06-14 02:40 | phase0a_arm_c | 0a norm arm C | leaky | 0.670 | ✅ |
| 6 | 06-14 04:09 | phase0c_seed42 | 0c baseline seed 42 | leaky | 0.790 | ✅ |
| 7 | 06-14 04:16 | phase0c_seed43 | 0c baseline seed 43 | leaky | 0.786 | ✅ |
| 8 | 06-14 05:35 | phase0c_seed44 | 0c baseline seed 44 | leaky | 0.797 | ✅ |
| 9 | 06-15 01:22 | phase2_scale_25 | 2 data-scale 25% | leaky | 0.764 | ✅ |
| 10 | 06-15 04:27 | phase2_scale_50 | 2 data-scale 50% | leaky | 0.772 | ✅ |
| 11 | 06-15 09:35 | phase2_scale_75 | 2 data-scale 75% | leaky | 0.792 | ✅ |
| 12 | 06-15 14:11 | phase3_loss_compound_2to1 | 3 loss sweep | leaky | 0.793 | ✅ |
| 13 | 06-15 14:11 | phase3_loss_tversky_3to7 | 3 loss sweep | leaky | 0.590 | ✅ |
| 14 | 06-15 14:20 | phase3_loss_tversky_2to8 | 3 loss sweep | leaky | 0.073 | ❌ collapsed |
| 15 | 06-15 14:26 | phase3_loss_compound_1to2 | 3 loss sweep | leaky | 0.800 | ✅ |
| 16 | 06-15 15:10 | phase3_loss_compound_1to1 | 3 loss sweep | leaky | 0.788 | ✅ |
| 17 | 06-16 06:33 | phase3_bd_focal_ignore_w1 | 3 boundary factorial | leaky | 0.787 | ✅ |
| 18 | 06-16 06:56 | phase3_loss_compound_1to2_seed43 | 3 loss seed-confirm | leaky | 0.787 | ✅ |
| 19 | 06-16 06:57 | **phase3_bd_focal_ignore_w2** | 3 boundary factorial | leaky | **0.805** | ✅ **winner** |
| 20 | 06-16 07:15 | phase3_bd_compound_1to2_ignore_w2 | 3 boundary factorial | leaky | 0.800 | ✅ |
| 21 | 06-16 08:00 | phase3_bd_focal_ignore_w3 | 3 boundary factorial | leaky | 0.797 | ✅ |
| 22 | 06-16 09:10 | phase3_bd_compound_1to2_ignore_w1 | 3 boundary factorial | leaky | 0.800 | ✅ |
| 23 | 06-16 14:37 | phase3_bd_compound_1to2_ignore_w3 | 3 boundary factorial | leaky | 0.802 | ✅ |
| 24 | 06-17 00:06 | phase5_arch_fpn | 5 arch sweep | leaky | 0.794 | ✅ |
| 25 | 06-17 01:52 | phase5_arch_manet | 5 arch sweep | leaky | 0.621 | ✅ |
| 26 | 06-17 02:05 | phase3_bd_focal_ignore_w2_seed44 | 3 winner seed-confirm | leaky | 0.805 | ✅ |
| 27 | 06-17 02:11 | phase3_bd_focal_ignore_w2_seed43 | 3 winner seed-confirm | leaky | 0.820 | ✅ |
| 28 | 06-17 04:25 | phase3_bd_compound_1to2_ignore_w3_seed43 | 3 runner-up confirm | leaky | 0.817 | ✅ |
| 29 | 06-17 10:13 | phase3_bd_compound_1to2_ignore_w3_seed44 | 3 runner-up confirm | leaky | 0.815 | ✅ |
| 30 | 06-15 16:13 | phase5_arch_deeplabv3plus | 5 arch sweep | leaky | 0.788 | 🔵 early-stop tail |
| 31 | 06-16 11:27 | phase5_arch_pspnet | 5 arch sweep | leaky | 0.729 | 🔵 early-stop tail |
| 32 | — | phase5_arch_segformer | 5 arch sweep | leaky | — | ❌ dropped (no run) |
| 33 | 06-17 11:48 | phase4_extra_rgb_baseline | 4 EXTRA control | corrected | 0.830 | 🔵 early-stop tail |
| 34 | 06-17 12:50 | **phase4_extra_full** (8-band) | 4 EXTRA | corrected | **0.876** | 🔵 early-stop tail |
| 35 | 06-18 06:39 | **phase4_extra_ndvi** | 4 EXTRA | corrected | **0.888** | ✅ **best single** |
| 36 | 06-18 06:48 | phase4_extra_tc (tasseled-cap) | 4 EXTRA | corrected | 0.868 | 🔵 early-stop tail |
| 37 | 06-18 07:49 | phase4_extra_nbr | 4 EXTRA | corrected | 0.847 | ✅ |
| 38 | 06-18 08:35 | phase4_extra_se_proto | 4 EXTRA | corrected | 0.847 | ✅ |
| 39 | 06-18 09:05 | phase4_extra_se_pca | 4 EXTRA | corrected | 0.874 | ✅ |
| 40 | 06-18 13:52 | phase10_curric_r30_pf50 | 10 curriculum (Step 2a) | corrected | 0.853 | ✅ |
| 41 | 06-18 13:52 | **phase10_curric_r20_pf33** | 10 curriculum (Step 2a) | corrected | **0.894** | ✅ best cell (s42) |
| 42 | 06-18 15:23 | phase10_curric_base | 10 curriculum control | corrected | 0.879 | ✅ |
| 43 | 06-18 | phase10_curric_r30_pf33 | 10 curriculum (Step 2a) | corrected | 0.853 | ✅ |
| 44 | 06-19 | phase10_curric_r20_pf33_seed43 | 10 curric seed-confirm | corrected | 0.901 | ✅ |
| 45 | 06-19 | phase10_curric_r20_pf33_seed44 | 10 curric seed-confirm | corrected | **0.859** | ✅ ⚠ low (variance) |
| 46 | 06-19 | phase4_extra_se_pca_seed43 | 4 EXTRA seed-confirm | corrected | 0.857 | ✅ (s42 was 0.874) |
| 47 | 06-19 | phase4_extra_ndvi_seed43 | 4 EXTRA seed-confirm | corrected | 0.8965 | ✅ |
| 48 | 06-19 | **phase4_extra_ndvi_seed44** | 4 EXTRA seed-confirm | corrected | **0.9111** | ✅ best single run |
| 49 | 06-19 | phase4_extra_full_seed43 | 4 EXTRA seed-confirm | corrected | 0.8619 | ✅ |
| 50 | 06-19 | phase4_extra_full_seed44 | 4 EXTRA seed-confirm | corrected | 0.8678 | ✅ |
| 51 | 06-19 | phase4_extra_ndvi_fastcheck | 0.1 stop-fix validation | corrected | 0.8934 | ✅ gate-neutral confirmed (≈ s42 0.888) |
| 52 | 06-20 | aug_ref | 3A aug control | corrected | 0.8661 | ✅ control |
| 53 | 06-20 | aug_p0_geom_only | 3A photometric-off | corrected | 0.7936 | ✅ −0.072 → photometric aug helps a lot |
| 54 | 06-20 | aug_p1_no_clahe | 3A drop CLAHE | corrected | 0.8541 | ✅ −0.012 (within noise) |
| 55 | 06-20 | **aug_scale_off** | 3B RandomScale off | corrected | **0.8862** | ✅ +0.020 → best aug arm |
| 56 | 06-20 | aug_p3_photo_x15 | 3A photometric ×1.5 | corrected | 0.8658 | ✅ ≈ ref (no extra gain) |
| 57 | 06-20 | aug_pad_ignore | 3B pad-ignore fix (fill_mask) | corrected | 0.8527 | ✅ −0.013 → pad fix ≠ the lever; downscale itself hurts |
| 58 | 06-20 | aug_ref_seed43 / seed44 | 3A aug-control seed-confirm | corrected | 0.8468 / 0.8808 | ✅ ref 3-seed mean 0.865 |
| 59 | 06-20 | **aug_scale_off_seed43 / seed44** | 3B drop-RandomScale confirm | corrected | 0.8673 / 0.8892 | ✅ **mean 0.881, +0.016 vs ref, 3/3 seeds → LOCKED drop** |
| 60 | 06-20 | phase4_extra_ndvi_seproto_seed43 / 44 | channel-sel ndvi+se_proto | corrected | 0.8988 / 0.8966 | ✅ mean 0.898 ≈ NDVI-alone (no gain) |
| 61 | 06-20 | phase4_extra_ndvi_sepca_seed43 / 44 | channel-sel ndvi+se_pca | corrected | 0.8949 / 0.9088 | ✅ mean 0.900 ≈ NDVI-alone (no gain) |
| 62 | 06-21 | phase4_extra_ndvi_nbr | greedy round-1 ndvi+nbr | corrected | 0.8559 | ✅ < NDVI-alone (no add) |
| 63 | 06-21 | phase4_extra_ndvi_tc | greedy round-1 ndvi+tc | corrected | 0.8996 | ✅ ≈ NDVI-alone (no add) |
| 64 | 06-20 | **deploy_v1_ndvi_seed42/43/44** | **I: final-lock 3-seed (v2 recipe)** | corrected | **0.9144 / 0.9068 / 0.9156** | ✅ mean **0.9123** (spread 0.907–0.916; +0.014 over NDVI-alone 0.8985 → boundary+aug additive) — **v2 reference baseline** |
| 65 | 06-21 | phase4_f3_full | D: F3 dual-encoder late fusion | corrected | ~0.78 (peak) | ✅ losing — ≪ NDVI-alone (still finishing; verdict set) |
| 66 | 06-21 | phase4_f5_full | D: F5 cross-modal attn (8-band) | corrected | ~0.83 (peak) | ✅ losing — ≪ NDVI-alone (still finishing; verdict set) |
| 67 | 06-21 | phase4_f5_ndvi_seproto | D: F5 cross-modal attn (pair) | corrected | ~0.87 (peak) | ✅ losing — < NDVI-alone |
| 68 | 06-21 | effb3_deploy | E: EffB3 capacity-down probe | corrected | 0.9050 | ✅ Δ−0.0072 vs EffB5 0.9123 → **no-win** (capacity isn't the lever; B5 stays; B3 a ~0.7%-cheaper deploy fallback if needed) |
| 69 | 06-21 | aug_copypaste_deploy | F: copy-paste screen | corrected | 0.8930 | ✅ Δ−0.0192 → **worst aug arm** (instance-paste breaks spatial-context/shadow cues) |
| 70 | 06-21 | aug_mosaic_deploy | F: mosaic screen | corrected | 0.9069 | ✅ Δ−0.0054 → no-win (within deploy seed spread) |
| 71 | 06-21 | aug_cutmix_deploy | F: cutmix screen | corrected | 0.9014 | ✅ Δ−0.0109 → no-win |
| 72 | 06-22 | aug_mixup_deploy | F: mixup screen | corrected | 0.9028 | ✅ Δ−0.0095 → no-win — **completes mixing-aug family at 4/4 no-win** |
| 73 | 06-22 | phase4_fm_dinov3_ndvi | D/E: web DINOv3+NDVI (fair foundation test) | corrected | 0.9120 | ✅ **ties EffB5+NDVI 0.9123 (Δ−0.0003)** — the DINOv3-RGB edge (0.873>0.830) **vanishes once NDVI is added**; web foundation ≯ EffB5 with NDVI |
| 74 | 06-22 | aug_trivialaugment_deploy | F: TrivialAugment (shadow-safe pool) | corrected | 0.9167 | ✅ Δ+0.0045 (sub-gate) → no-win; see #80/#80b (3-seed F-closure) |
| 75 | 06-22 | aug_randaugment_deploy | F: RandAugment num_ops=2 (shadow-safe pool) | corrected | 0.9089 | ✅ Δ−0.0034 → no-win; see #79 |
| — | — | aug_anneal_deploy (+seed43/44) | F: aug-strength annealing 3-seed | corrected | done | ✅ ran (feat/aug-anneal, merged); no-win vs deploy 0.9123 — F closed |
| 76 | 06-22 | fm_sam2_rgb | E: SAM2/Hiera foundation encoder, RGB-only | corrected | **0.590** | ✅ **non-competitive** (≪ EffB5-RGB 0.830); hierarchical Hiera features weak for RTS |
| 77 | 06-22 | **phase4_fm_dinov3_ndvi** | D/E: web-DINOv3+NDVI (fair foundation test) | corrected | **0.9120** | ✅ **ties EffB5+NDVI 0.9123** (best_epoch 40) — generic web foundation is NOT the lever once NDVI is added |
| 78 | 06-22 | aug_mixup_deploy | F: mixup screen | corrected | 0.9028 | ✅ Δ−0.0095 → no-win (mixing-aug family now 4/4 struck out) |
| 79 | 06-22 | aug_randaugment_deploy | F: RandAugment (shadow-safe pool) | corrected | 0.9089 | ✅ Δ−0.0034 → no-win (best_smoothed; peak 0.914 was a single-epoch tail) |
| 80 | 06-22 | **aug_trivialaugment_deploy** | F: TrivialAugment (shadow-safe pool) | corrected | **0.9167** | ✅ **Δ+0.0045** (best_smoothed) → best aug arm but **below G=0.0112** → 3-seed confirm launched (seed43/44) to close F: noise or small real gain? |
| 80b | 06-22 | aug_trivialaugment_deploy_seed43/44 | F: TrivialAugment seed-confirm | corrected | 0.9216 / 0.9270 | ✅ 3-seed (with #80 0.9167) **mean 0.9218, 3/3 > deploy 0.9123** but Δ+0.0095 **sub-gate (<G)** → best aug arm, **not locked** (F-closure) |
| 81 | 06-22 | **fm_dinov3sat_l_rgb** | E: satellite-DINOv3 ViT-L (SAT-493M), RGB-only | corrected | **0.9320** (seed42) | ✅ **+0.020 over deploy 0.9123 on RGB-ONLY → BREAKOUT.** v2 encoder bet (user go 06-22); seed43/44 ran (numbers to harvest in close-out) |
| 82 | 06-22 | **fm_dinov3sat_l_ndvi** | E: satellite-DINOv3 ViT-L + NDVI | corrected | **0.9274 / 0.9199 / 0.9198** | ✅ 3-seed mean **~0.9224** (logged smoothed peaks; runs terminated post-peak 06-24) — **+0.010 over deploy 0.9123** (just under G); big object-level gains (IoU_rts ~0.69 vs 0.56, obj-F1 ~0.62 vs 0.40). seed42 best ckpt lost to the resume-clobber bug → retrain for a clean deploy set |
| 83 | 06-22 | fm_dinov3_rgb_imagenet | E control: web-DINOv3 RGB + native ImageNet norm | corrected | 0.884 | ✅ de-confounds norm: ImageNet-norm 0.884 > z-score 0.873 → norm matters; still ≪ sat ViT-L |
| 84 | 06-22 | ❌ fm_dinov3sat_7b_frozen | E: satellite-DINOv3 7B frozen linear-probe | corrected | 0.498 (peak) | ❌ **killed** — diverged (constant frozen_lr=1e-3, no anneal; val 0.498→0.0028) AND non-competitive; frozen sat-7B features need fine-tuning |
| 85 | 06-22 | ablation_noignore_ndvi_seed42/43/44 | C: **ignore-region ablation** (train-only, no manual ignore) | corrected | — | ⏸ PAUSED — needs a separate RTS-truth source (positives overwrite ignore); code on `main`, ablation deferred |

---

## Working branches & worktrees

**Consolidated to only-`main` (2026-06-24).** The v2 campaign is closed and every branch/worktree above is
merged: `feat/aug-anneal`, `report/overhaul` (this ledger + `build_report.py` overhaul), and
`ablation/no-ignore-regions` (label-regen code; ablation itself PAUSED — needs an RTS-truth source) all landed
on `main`; the aug-anneal worktree was removed and `origin/feat/aug-anneal` deleted. No active worktrees remain.
The only live training is the **DINOv3+NDVI seed42 retrain** (`fm_dinov3sat_l_ndvi_seed42_rerun`, replacing the
checkpoint lost to the resume-clobber bug). Forward work tracked in `.claude/plans/elegant-exploring-lemur.md`
("from now → start of full inference"): Phase C NDVI-at-inference reader is built; Phase D = calibrate +
solo-vs-ensemble select on Val → Test-Realistic once → package.

## Queued / live training

**Campaign run-complete (2026-06-24) — queue drained; all GPUs idle except the one retrain below.** Every
formerly-queued experiment ran (sat-DINOv3 RGB/+NDVI seed43/44 → master #81/#82/#106; aug-anneal 3-seed → #74
note; the earlier aug arms → #55–57). The auto-dispatchers (`autolaunch_satconfirm.sh` etc.) have exited.

| Live now | GPU | Notes |
|----------|-----|-------|
| **fm_dinov3sat_l_ndvi_seed42_rerun** | 0 | Re-training the +NDVI seed42 whose `best_deployment.pth` was lost to the resume-clobber bug (now fixed) → restores a clean 3-seed `+NDVI` deploy set for the Phase-D Val compare. |

**Not run (deferred by decision — do NOT gate v2):** Stage-0.2 bootstrap 1:50/1:100 high-ratio metric ·
Stage-0.3 v1.0 re-stage (+28 pos / −49 black) · MAE SSL · hard-neg mining · the optional C3 ignore_w re-confirm.

---

## Program status — planned / in-progress / conditional / dropped + discussed-but-didn't-land

Model = **v2** (repo RTSmapping_v2); first pan-Arctic map = v2 product; re-stage / hard-neg / MAE → v3+.
Plan: `.claude/plans/elegant-exploring-lemur.md`. Family scheme (replacing the old §N/Stage-N mix):
**A** Baseline · **B** Data · **C** Loss/boundary · **D** Channels/fusion · **E** Architecture/encoder ·
**F** Augmentation · **G** Sampling · **H** Calibration/TTA · **I** Final-lock/Test · **J** Deploy/inference · **K** Deferred.

**✅ Locked:** A (μ₀=0.7912, G=0.0112) · B (data plateau) · C (focal·ignore_w2) · D (**EXTRA=NDVI** + **F0** channel-stack) ·
F (drop-RandomScale; keep photometric+CLAHE) · G (default sampling). Infra: `base_v2_fast` stop-fix · gate policy (mean Δ≥G **and** 3-seed sign-consistency).

**🟢 Encoder verdict (E) — RESOLVED 2026-06-24:** decoders all lose to UNet++ and EffB3 capacity-down is no-win, **but the satellite-pretrained ViT-L is the campaign's biggest win.** sat-DINOv3 ViT-L + NDVI 3-seed **~0.9224** (0.9274/0.9199/0.9198) vs EffB5+NDVI **0.9123** — +0.010 PR-AUC (near-miss on G) but **decisive on object metrics: pixel-IoU 0.64 vs 0.51 (+0.13), obj-F1 0.56 vs 0.39 (+0.17)**. sat-RGB ties +NDVI on PR-AUC (~0.913, 3 clean ckpts). Generic web-DINOv3+NDVI only ties EffB5 (0.9120); SAM2/Hiera 0.590 + 7B-frozen 0.498 non-competitive. **The E (deploy-encoder) choice supersedes EffB5 → sat-DINOv3 ViT-L is the leading deploy**; user opted to **include NDVI** and pick solo-vs-ensemble (vs EffB5+NDVI) on calibrated Val at the final lock (I, pending). **PRs #42 (sat encoder) + #26 (S2/EXTRA doc) merged.**

**🔵 Live:** only **fm_dinov3sat_l_ndvi_seed42_rerun** (GPU0) — restoring the +NDVI seed42 deploy checkpoint lost to the resume-clobber bug. Everything else is run-complete.

**✅ Screens all landed (no-win vs deploy 0.9123):** EffB3 0.9050 · copy-paste 0.8930 · mosaic 0.9069 · cutmix 0.9014 · mixup 0.9028 (**mixing-aug family 4/4 struck out**) · aug-anneal 3-seed mean 0.916 · **TrivialAugment 3-seed mean 0.9218** (best aug arm, 3/3 positive but Δ+0.0095 **sub-gate**) · RandAugment 0.9089 — F closed, none earns a lock. SAM2 0.590 + 7B-frozen 0.498 non-competitive.

**⏳ To-do before v2 ship (forward plan `.claude/plans/elegant-exploring-lemur.md`):** finish the seed42 retrain → **Phase D**: H calibration (temp + threshold + D4-TTA) on Val + the RGB/+NDVI/ensemble select → **Test-Realistic once** → package. Phase C NDVI-at-inference reader is **built**. Then Phase E inference infra (quad cache, bucket, fleet) → Phase F pre-flight → full inference.

**⏸️ Conditional (gated on a trigger):** H scale-TTA (scale-transfer test) · J hard-neg mining (post first inference) · K MAE (user-go; end-stage parallel w/ inference) · context-expansion multi-scale (post-inference map review) · val-negative growth (if bootstrap readout becomes decisive) · ensemble (decide at final lock).

**📦 Deferred to v3 (post-v2-ship):** re-stage (+28 pos / −49 black) · hard-negative mining · MAE (if not run in the v2 window).

### ❌ Dropped & 💭 discussed-but-didn't-land (record of what we considered and why it isn't in v2)
| Idea | Fam | Verdict / why |
|---|---|---|
| §6.5 loss×wd×curriculum interaction check | C/F/G | **dropped** (per user) — moot after wd dropped + curriculum rejected + loss locked |
| wd × aug regularization grid (§6.3) | F | **dropped** — over-parameterization trigger never fired (gap 0.05/0.17 < 0.4); wd=5e-2 untested |
| Curriculum r20_pf33 | G | **tested → rejected** — within seed noise (0.894/0.901/0.859) |
| RandomScale downscale aug | F | **tested → dropped** — 3-seed A/B: removing it +0.016 (all seeds) |
| F2 channel-attention (full 8-band) | D | **tested → collapsed** (0.827) |
| F3 dual-encoder / F5 cross-modal attn | D | **tested → lose to F0** (≪ NDVI-alone) — heavy fusion extracts less than the stack |
| Mixing augs: copy-paste / mosaic / cutmix / mixup | F | **tested → no-win** (06-22; 0.893 / 0.907 / 0.901 / running vs deploy 0.9123) — copy-paste worst (breaks spatial-context/shadow cues) |
| EffB3 capacity-down | E | **tested → no-win** (0.9050, Δ−0.007 vs EffB5) — capacity isn't the lever; kept as a cheaper deploy fallback only |
| RandAugment / TrivialAugment | F | **running 06-22** (user-revived after a brief evidence-based deprioritization) — auto-policy over a shadow-safe pool; gate vs deploy 0.9123 |
| Aug-strength annealing | F | **TO-DO (queued 06-22 PM)** — needs epoch-aware transform plumbing; building it now (off-by-default, strong→mild by ~ep40) then 3-seed vs deploy 0.9123. (Earlier "deprioritized" note was Claude's audit call, **not** the user's — corrected.) |
| SegFormer (mit_b5) | E | **dropped** — low value on a plateau; foundation is the better transformer bet |
| EffB7 | E | **dropped** — overfit risk on a plateau (bound-only) |
| UNet3+ | E | **dropped** — condition unmet (no decoder family moved the gate) |
| YOLO / instance-seg (Mask R-CNN) | E | **rejected** — paradigm mismatch (coarse proposal-anchored masks) |
| SAM3 | E | **dropped** — image incompatible (py3.12/torch2.7); SAM2 used instead |
| DINOv3 + EXTRA (earlier "dropped") | E | **revived** — the earlier drop used an unfair comparison (DINOv3-RGB 0.873 *beat* EffB5-RGB 0.830); now a to-do |
| Re-run Phase 2 on 3500 positives | B | **moot** — no new labels coming |
| Pseudo-labeling / self-training | K | **backup only** — confirmation-bias risk; revisit only if representation gains plateau |
| Soft-label boundary handling | C | **deferred / not implemented** (data.md) — `ignore` covers annotation noise for v2 |

---

## Cluster takeaways (best-in-cluster)

- **Loss (Phase 3, leaky):** focal + `ignore_w2` is the boundary winner (0.805, seed-confirmed
  0.805–0.820). compound_1to2 + `ignore_w3` close runner-up (0.802–0.817). Tversky variants collapse.
- **Architecture (Phase 5, leaky):** UNet++ baseline (0.790) ≥ FPN (0.794) > DeepLabV3+ (0.788) >
  PSPNet (0.729) ≫ MANet (0.621). No CNN decoder beats UNet++ → arch stays UNet++/EffB5.
- **EXTRA channels (Phase 4, corrected):** NDVI-alone (0.888) ≈ full 8-band (0.876) ≈ SE-PCA (0.874)
  ≫ RGB control (0.830). NBR/SE-proto (+0.017) are weak. → NDVI is the efficient ceiling; the open
  question (Step 3) is whether a channel **combination** + better **fusion** beats NDVI-alone.
  **Seed-confirmed (3 seeds, final):** NDVI 0.888 / 0.8965 / 0.9111 → **mean 0.8985, std 0.0095**;
  full 8-band 0.876 / 0.8619 / 0.8678 → mean 0.869, std 0.007. NDVI beats RGB by ~0.07 (≫ σ) and beats
  full by ~0.03 → NDVI is a **real, low-variance win** and the **efficient channel**.
- **🔒 Channel selection — greedy forward from NDVI COMPLETE (F0 early-stack, corrected):** anchor
  NDVI-alone **0.8985**. All round-1 additions: +se_pca **0.900**, +se_proto **0.898**, +tc **0.900**, +nbr
  **0.856** — none clears the gate (all ≤ anchor, within σ) → **greedy terminates, no channel added →
  LOCKED EXTRA = `[NDVI]`** (RGB+NDVI, 4-channel F0 stack).
- **🔒 Fusion (D) — F0 early channel-stack LOCKED, now evidence-based:** light fusion F0/F1/F2 all tie
  NDVI-alone; **heavy fusion F3/F5 LOSES** — F3-full ~0.78, F5-full ~0.83, F5-pair ~0.87 (all ≪ 0.8985,
  below even F0). Dual-encoder / cross-modal attention extract *less* than the simple channel-stack here →
  the skip-condition is confirmed, not assumed. DINOv3+NDVI (fair encoder test) still to run (family E).
- **🔒 Final-lock (I) — v2 deploy recipe, 3-seed:** RGB+NDVI · F0 · focal·ignore_w2 · default sampling ·
  aug−RandomScale · base_v2_fast → **0.9144 / ~0.916 / 0.9156 (mean ~0.915)**. That's **+0.017 over
  NDVI-alone (boundary none) 0.8985** → boundary-ignore + drop-RandomScale are **additive on top of NDVI**,
  tight across seeds. Pre-ship screens (EffB3, mixing augs, SAM2, DINOv3+NDVI, calibration) run before the
  one-shot Test-Realistic; recipe re-locks only if a screen earns it.
- **Curriculum (Phase 10, corrected):** r20_pf33 best cell single-seed 0.894, **but seed-confirm is
  high-variance: 0.894 / 0.901 / 0.859 → mean ≈0.885 vs base 0.879 (Δ≈0.006), within std ≈0.021.**
  The curriculum "win" is **not distinguishable from seed noise** at 3 seeds — treat as unconfirmed.
- **🔒 Gate vs measured variance (RESOLVED 2026-06-21):** measured corrected-split seed std ranges
  ~0.007–0.021 (NDVI 0.0095, full 0.007, aug_scale_off 0.012, aug_ref 0.017, curriculum 0.021) → σ_corrected
  ≈ 0.012, **~2× the leaky σ₀=0.0056** behind G=0.0112. **Policy: keep G=0.0112 as a single-seed SCREEN, but
  every LOCK requires a 3-seed confirm judged on BOTH (a) mean Δ ≥ G AND (b) sign-consistency across all 3
  seeds.** Sign-consistency is the decisive test: drop-RandomScale (+0.016 mean, **3/3 positive**) → locked;
  curriculum r20_pf33 (+0.006 mean, sign **flipped** s44) → rejected as noise. This is the discipline already
  applied to every second-wave lock; NDVI's ~0.07 margin clears it trivially.
- **Stop-schedule fix (Stage 0.1, audit 2026-06-19):** all 48 prior runs peaked by ~ep52 then trained a
  median 40 wasted epochs (41% of GPU-h, overfitting tail), best checkpoint unchanged. New `base_v2_fast`
  (patience 8→5, start_epoch 101→45, max_epochs 300→120) is **gate-neutral** — validated by `fastcheck`
  (0.8934 ≈ original NDVI 0.888). ~2× throughput; all second-wave runs inherit it.
- **Augmentation study (Stage 3A/3B, corrected, single-seed vs control aug_ref 0.866):** **(1) photometric aug
  matters** — geometric-only craters to 0.794 (−0.072); dropping CLAHE −0.012 and ×1.5 photometric ≈0 (within
  noise) → keep the current photometric set, don't strengthen it. Consistent with PlanetScope basemap RGB being a
  CV-optimized visual product. **(2) RandomScale downscale HURTS** — `aug_scale_off` 0.886 (**+0.020, best arm**)
  > control; and `aug_pad_ignore` (scale on, pad-border bug fixed) 0.853 is *below* the buggy control → the lever
  is the **downscale aug itself**, not the pad-ignore labeling. **🔒 3-seed A/B confirms it:** aug_scale_off
  0.886/0.867/0.889 (**mean 0.881**) vs aug_ref 0.866/0.847/0.881 (**mean 0.865**) → **Δ+0.016, positive in all
  3 seeds** → **DROP RandomScale from the locked recipe** (photometric set + CLAHE kept).
