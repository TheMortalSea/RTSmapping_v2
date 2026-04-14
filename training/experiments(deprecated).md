## 13. Experiment Tracking

### 13.1 MLflow Configuration

**Tracking URI**: GCS-backed MLflow at `gs://abruptthawmapping/mlflow/`. Configurable via YAML:
```yaml
mlflow:
  tracking_uri: "gs://abruptthawmapping/mlflow/"
  experiment_name: "rts-segmentation-v2"
```
The `MLFLOW_TRACKING_URI` environment variable overrides the YAML value if set (for flexibility).

**Experiment Structure**:
- Experiment name: `rts-segmentation-v2`
- Each run includes: hyperparameters, metrics, artifacts, system info

**Required Parameters to Log**:

| Category | Parameters |
|----------|------------|
| Model | architecture, backbone, pretrained, input_channels, input_size |
| Loss | loss_function, focal_gamma, focal_alpha, boundary_ignore_width |
| Optimizer | optimizer, learning_rate, weight_decay, gradient_clip_norm |
| Schedule | scheduler, warmup_epochs, min_lr, freeze_backbone_epochs |
| Training | batch_size, max_epochs, early_stopping_patience, ema_decay |
| Data | data_version, train_pos_neg_ratio, curriculum_schedule |
| System | git_commit, pytorch_version, cuda_version, gpu_model, gpu_count |

**Metrics to Log Per Epoch**:
- train_loss, train_iou_rts
- val_balanced_iou, val_balanced_pr_auc
- For each ratio (200, 500, 1000): val_{ratio}_pr_auc, val_{ratio}_iou_rts, val_{ratio}_obj_precision, val_{ratio}_obj_recall

**Artifacts to Save**:
- best_model.pth (EMA weights)
- normalization_stats.json
- config.yaml
- pr_curves.png
- threshold_calibration.json
- requirements_frozen.txt
- predictions.png (fixed 3 positive and 3 negative validation images subplot 3 columns by 2 rows)

### 13.2 Experiment Progression

Experiments should follow dependency order:

**Phase 1: Baseline**
- RGB, UNet3+, EfficientNet-B7, Focal loss
- Establish baseline performance

**Phase 2: Loss Ablation** (depends on Phase 1)
- Compare: Focal, Tversky (β > α), Class-balanced CE
- Select best loss function

**Phase 3: Curriculum Ablation** (depends on Phase 2)
- Compare: Fixed ratios (1:10, 1:20), Curriculum schedules
- Select best imbalance handling

**Phase 4: Architecture** (depends on Phases 2-3)
- Compare: UNet3+, DeepLabV3+, SegFormer, SAM fine-tuned
- Select best architecture

**Phase 5: Auxiliary Data** (depends on Phase 4)
- Compare: RGB only, RGB+NDVI, RGB+DEM, RGB+all EXTRA channels
- Determine if auxiliary data helps

**Phase 6: Fusion Method** (only if Phase 5 shows benefit)
- Compare: Early fusion, Late fusion
- Select best fusion strategy

**Experiment execution**: A single `scripts/train.py` handles all experiments. Each experiment is defined by its own YAML config file in `configs/`:
```
configs/
├── baseline.yaml         # Phase 1: UNet3+, focal loss
├── exp02_loss.yaml       # Phase 2: loss ablation (focal vs tversky vs class-balanced CE)
├── exp03_curriculum.yaml # Phase 3: curriculum schedule ablation
├── exp04_arch.yaml       # Phase 4: architecture comparison
└── exp05_aux.yaml        # Phase 5: auxiliary data ablation
```
Run an experiment: `python scripts/train.py --config configs/baseline.yaml`
---
