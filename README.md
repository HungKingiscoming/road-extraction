# DualBranchRoadNet for Road Extraction from Remote Sensing Imagery

This repository implements a PyTorch road extraction pipeline for high-resolution aerial and satellite imagery. The project focuses on binary road segmentation and is evaluated on two standard benchmarks: Massachusetts Roads and DeepGlobe Road Extraction.

The code is built from scratch with plain PyTorch and does not depend on segmentation libraries such as `mmsegmentation`, `segmentation_models_pytorch`, or `albumentations`.

## Highlights

- Road extraction as binary semantic segmentation
- Dual-resolution architecture with a detail stream and a semantic context stream
- DAPPM-style multi-scale context aggregation
- Centerline-aware auxiliary supervision for topology preservation
- Native-resolution inference with sliding-window blending
- DDP training support and checkpoint resume
- Re-parameterizable blocks for deployment efficiency

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── train.py
├── test_native.py
├── compare_reparameterization.py
├── run-gpu-container.sh
├── modeling/
│   ├── __init__.py
│   ├── decoder.py
│   └── model.py
└── road-extraction-main/   # duplicate folder present in the workspace
    ├── requirements.txt
    ├── run-gpu-container.sh
    ├── train.py
    └── modeling/
```

## Problem setting

Given an RGB image $I \in \mathbb{R}^{H \times W \times 3}$, the model predicts a binary road mask $\hat{Y} \in \{0,1\}^{H \times W}$ where road pixels are labeled as 1. This is a highly imbalanced segmentation task because roads usually occupy a small fraction of the image, while roads are also thin, elongated, and structurally important.

The implementation addresses this with:

- class-weighted cross-entropy
- Dice loss for region quality
- centerline Tversky supervision to preserve connectivity
- sliding-window inference at native resolution
- residual context fusion between shallow detail features and deep semantic context

## Installation

### Requirements

- Python 3.10+
- PyTorch with CUDA support recommended for training
- `torchvision`
- NumPy, Pillow, tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

> The project expects a CUDA-compatible PyTorch + torchvision pair. On Kaggle or similar GPU environments, avoid reinstalling the bundled torch packages unless you know the environment is compatible.

## Dataset preparation

### Massachusetts Roads

Typical layout:

```text
/data_root/
├── images/
├── labels/
├── train.txt
├── test.txt
```

The split follows the order in the text files. The script explicitly checks for overlap between training and evaluation data.

### DeepGlobe Road Extraction

Typical layout:

```text
/data_root/
├── train/
├── valid/
├── test/
```

The project supports a deterministic overlapping split protocol for reproducibility. It writes a `split_manifest.json` file for training and evaluation alignment.

## Training

### Single-GPU training on Massachusetts

```bash
python train.py \
  --dataset massachusetts \
  --data_root /path/to/massachusets \
  --train_list /path/to/massachusets/train.txt \
  --test_list /path/to/massachusets/test.txt \
  --epochs 200 \
  --crop_size 1024 \
  --batch_size 2 \
  --accumulation_steps 2 \
  --lr 2e-4 \
  --bilateral_fusion spatial \
  --road_occlusion_probability 0.3 \
  --save_dir ./checkpoints/mass_dualbranch
```

### DeepGlobe training

```bash
python train.py \
  --dataset deepglobe \
  --data_root /path/to/deepglobe \
  --deepglobe_train_count 5000 \
  --deepglobe_val_from_test_count 300 \
  --epochs 120 \
  --crop_size 1024 \
  --save_dir ./checkpoints/dg_dualbranch
```

### Multi-GPU DDP

```bash
torchrun --nproc_per_node=2 train.py --dataset massachusetts --data_root /path/to/data
```

### Resume training from checkpoint

```bash
python train.py --dataset massachusetts --resume ./checkpoints/mass_dualbranch/last.pt
```

### Transfer learning with progressive unfreezing

```bash
python train.py --dataset deepglobe \
  --pretrained_checkpoint ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --transfer_weights ema
```

## Model architecture

The core model is `DualBranchRoadNet`, built on a truncated ResNet-34 encoder.

Main design principles:

- Keep a high-resolution detail stream at S8
- Use a semantic stream at S16/S32 to gather long-range context
- Fuse semantic context back into the detail path with residual gates
- Reconstruct the road mask from multi-scale features
- Add an auxiliary centerline head during training for connectivity supervision

The network is designed to preserve thin road structures while still using large receptive fields to distinguish roads from visually similar linear patterns such as rivers, shadows, fences, and rooftops.

## Loss function

Training combines:

- weighted cross-entropy for class imbalance
- Dice loss for region coverage
- centerline Tversky loss for topological connectivity

This helps the model recover thin roads and maintain continuity between distant segments.

## Evaluation

The project evaluates at native image resolution instead of resizing inputs. It uses an overlapping sliding-window strategy with Hann-weighted logit blending to reduce seam artifacts.

### Validation / testing

```bash
python test_native.py \
  --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset val61 \
  --search-threshold
```

Then apply the threshold on the test split:

```bash
python test_native.py \
  --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset test117 \
  --thr 0.46 \
  --tta-mode flip4 \
  --out cache_test117.npz
```

DeepGlobe evaluation:

```bash
a python test_native.py \
  --ckpt ./checkpoints/dg_dualbranch/best_fixed_road_iou.pt \
  --subset deepglobe_test1226
```

### Main metrics

- road IoU (primary metric)
- background IoU
- mIoU
- precision, recall, F1
- macro IoU over images
- relaxed F1 with a small spatial tolerance for topology quality

## Checkpoints and outputs

Each run saves a checkpoint directory such as:

```text
save_dir/
├── best_fixed_road_iou.pt
├── best_calibrated_road_iou.pt
├── last.pt
├── split_manifest.json
├── metrics.jsonl
└── ...
```

The checkpoint stores the model, EMA weights, optimizer state, scheduler state, scaler state, epoch, validation metrics, and full training arguments, allowing exact resumption and reproducible evaluation.

## Deployment optimization

The network includes re-parameterizable blocks (`RepVGGBlock` and `RepDepthwiseBlock`), allowing training-time multi-branch structures to be fused into a single convolutional representation for inference.

```python
model.switch_to_deploy()
```

The script `compare_reparameterization.py` verifies that the deploy-time model matches the training-time model within numerical tolerance and reports throughput and memory efficiency.

## Training notes

- `batch_size` is per-GPU
- `accumulation_steps` controls effective batch size
- AMP (automatic mixed precision) and `channels_last` are enabled by default
- EMA weights are used for validation and testing
- progressive unfreezing is supported for transfer learning scenarios

## Useful scripts

- `train.py`: training pipeline
- `test_native.py`: native-resolution validation/test inference
- `compare_reparameterization.py`: deployment equivalence and efficiency comparison
- `modeling/model.py`: model architecture
- `modeling/decoder.py`: loss and decoder logic

## Citation / project context

This project is intended for research and experimentation in remote sensing road extraction and is not a general-purpose geospatial production package. It is best suited for reproducible academic experiments and controlled benchmark evaluation.

## License

This repository does not include an explicit license file in the workspace snapshot. Please check the project source or institutional repository policy before reuse or redistribution.


$$\text{imbalance} = \frac{N_{\text{background}}}{N_{\text{road}}}, \qquad w_{\text{road}} = \min\!\left(\sqrt{\text{imbalance}},\ \text{cap}\right)$$

with `--road_weight_cap` defaulting to 2.0. Rank 0 scans every training mask and
broadcasts the result to the other ranks. Skip the scan entirely with
`--fixed_road_weight`.

**(b) Binary Dice** on the road probability — stabilizes gradients when the positive
class is rare.

**(c) Centerline Tversky** — the main topological contribution:

- **Centerline targets are generated without any external library:**
  `soft_skeletonize` performs morphological thinning using only `max_pool2d`
  (erode = `-maxpool(-x)`, dilate = `maxpool(x)`, open = dilate∘erode), iterating
  `--skeleton_iterations` times (default 8).
- **Skeletonize BEFORE downsampling** — max-pooling the mask straight down to S4
  would merge narrow branches and intersections.
- Dilate slightly (`--centerline_dilation 1`), then `adaptive_max_pool` to S4 to
  match the auxiliary head.
- Tversky with $\alpha = 0.30 < \beta = 0.70$ **penalizes false negatives more
  heavily**, i.e. it penalizes **broken roads** more than spurious ones — exactly the
  connectivity objective.
- **Ramp-up schedule:** disabled entirely before `--aux_start_epoch` (5), then ramped
  linearly over `--aux_warmup_epochs` (5) up to `--aux_weight` (0.15). The centerline
  branch is only meaningful once the coarse mask is roughly correct.
- `--fast_centerline_target` skeletonizes at an intermediate S2 resolution (cheaper),
  rescaling the iteration count and dilation radius accordingly.

---

## 5. Training Strategy

### 5.1. Road-guided cropping (`random_crop_pair`)

Uniform random crops over images with only 2–5% road pixels produce many **completely
empty** crops. Instead, with probability `--road_crop_probability` (0.60):

1. Max-pool the mask down 8× (`coarse_max_mask`) to a grid of road-containing cells.
2. Sample a road cell at random and jitter the offset so the road is not always
   centered.
3. Try up to `--road_crop_tries` (8) times, keeping the crop with the highest road
   fraction, and stop early once `--road_crop_min_fraction` (0.002) is reached.

Images smaller than the crop are **reflect-padded** while masks are **zero-padded**.

### 5.2. Augmentation

- **Label-preserving:** horizontal/vertical flips plus 90° rotations (the D4 group) —
  valid because remote sensing imagery has no privileged "up" direction.
- **Photometric:** brightness/contrast (p=0.60), saturation (p=0.35), Gaussian blur
  (p=0.15), Gaussian noise (p=0.15).
- **`road_guided_occlusion` (domain-specific, `--road_occlusion_probability`).** Draws
  occluders *directly on labeled road pixels*: shadows (brightness 0.30–0.65),
  vegetation (green patches), or vehicles (bright patches). **The segmentation target
  is deliberately left unchanged**, forcing the model to *infer* the hidden road
  segment from surrounding continuity. Occluders are kept small (0.6–1.8% of the
  shorter side) so the reconstruction problem stays solvable.

### 5.3. Optimization

- **AdamW** (fused when CUDA is available). `--weight_decay 1e-4` applies only to
  tensors with `ndim > 1`; biases, norm parameters and gate scales get **no** weight
  decay.
- **Layered learning rates** (multiplied by `--lr`, default 2e-4):

  | Group | Factor | Default value |
  |---|---|---|
  | `head` (decoder) | 1.00 | 2.0e-4 |
  | `dual_branch` | `--dual_branch_lr_factor` 0.75 | 1.5e-4 |
  | `layer3`+`layer4`, `layer2` | `--backbone_lr_factor` 0.20 | 4.0e-5 |
  | `stem`+`layer1` | `--early_encoder_lr_factor` 0.10 | 2.0e-5 |

  `build_optimizer` **verifies that no parameter is duplicated or left out**.
- **LR schedule:** linear warmup over `--warmup_epochs` (5) starting from factor 0.10,
  then cosine annealing down to `--min_lr_ratio` (0.02). Stepped **per optimizer
  update**, not per epoch.
- **AMP fp16 + GradScaler**, `channels_last`, `cudnn.benchmark`, TF32. When GradScaler
  skips an update (non-finite gradients), the scheduler and EMA **do not step either**,
  keeping everything exactly in sync.
- **Gradient accumulation** via `--accumulation_steps` (2). Effective batch =
  `batch_size × world_size × accumulation_steps` (2×1×2 = 4 by default).
- **Gradient clipping** at norm 3.0. **EMA** with decay 0.999 and a ramp
  $d_t = 0.999\,(1 - e^{-t/2000})$ — **the EMA weights are what gets validated and
  tested**.

### 5.4. Progressive unfreezing

Enabled automatically for transfer runs (`--pretrained_checkpoint`) and disabled when
training from scratch; override with `--progressive_unfreeze` /
`--no-progressive_unfreeze`.

| Phase | Name | Default epochs | Trainable modules |
|---|---|---|---|
| 0 | `head_only` | 0–2 | decoder only |
| 1 | `head_plus_dual_branch` | 3–5 | + `dual_branch` |
| 2 | `plus_resnet_layer3_layer4` | 6–11 | + `layer3`, `layer4` |
| 3 | `plus_resnet_layer2` | 12–… | + `layer2` |
| 4 | `all_trainable` | `--unfreeze_all_epoch` (default **-1** = never) | + `stem`, `layer1` |

Frozen sections are wrapped in `torch.no_grad()` **inside `forward` itself**, which
saves activation memory rather than merely blocking gradients.
`enforce_frozen_norm_eval()` keeps the BatchNorm layers of frozen sections in `eval`
mode so their running statistics cannot drift (`--freeze_encoder_bn` defaults to on
for the **entire** encoder).

> The optimizer, scheduler and DDP reducer are always built while **every parameter is
> still trainable** (`set_trainable_phase(4)` beforehand), so later phase changes never
> invalidate a parameter group or a DDP hook.

### 5.5. Distributed training (DDP)

- NCCL with `init_method="env://"` and a **15-minute** timeout (rank 0 may scan several
  thousand masks before the first collective).
- `find_unused_parameters` is enabled only when progressive unfreezing is active.
- Backward runs on **every rank unconditionally** — DDP propagates non-finite
  gradients and GradScaler then skips the update consistently on all ranks. This
  removes one blocking all-reduce and two device synchronizations from **every healthy
  batch**.
- In-epoch metrics are gathered with **one** device-to-host copy covering five scalars.
- Validation uses `DistributedEvalSampler`, which shards **exactly** rather than
  padding with duplicated images the way `DistributedSampler` does (duplicates would
  corrupt the pooled IoU).
- `ema.module` is broadcast from rank 0 before each validation pass.

---

## 6. Evaluation

### 6.1. Native-resolution inference

Images are never resized. A sliding window of `--val_tile_size` (1024) with
`--val_overlap` (256) gives stride 768. Each tile is multiplied by a **2-D Hann
window** (clamped to a minimum of 0.05) and accumulated in the **logit domain**, then
divided by the accumulated weights — removing tile seams completely. Images smaller
than one tile are `reflect`-padded.

### 6.2. Metrics

All metrics are computed from the **dataset-pooled confusion matrix**, reported with a
`fixed_` prefix (threshold 0.5) and a `calibrated_` prefix (optimal threshold):

- `road_iou` — the **primary metric**, IoU of the road class alone.
- `background_iou`, `miou`, `precision`, `recall`, `f1`, `accuracy`.
- `fixed_road_iou_macro` — IoU averaged **per image** (macro).
- `fixed_relaxed_f1` — **relaxed F1 at ±3 px** (`--relaxed_buffer_px`): a prediction
  counts as correct if it falls within a 3 px dilation of the ground truth, and vice
  versa. This reflects **route/topology quality** and is far less sensitive to the
  1–2 px jitter of hand-drawn labels.

**Threshold calibration.** A 1001-bin histogram of probabilities per true class is
accumulated, then thresholds from 0.20 to 0.80 in steps of 0.02 are swept to maximize
road IoU — **without ever storing full probability maps**. The result is reported as
`calibrated_threshold`.

### 6.3. Run outputs

```
save_dir/
├── best_fixed_road_iou.pt        # best IoU @0.5
├── best_calibrated_road_iou.pt   # best IoU at the calibrated threshold
├── last.pt                       # for exact resumption
├── split_manifest.json           # split reproducibility
└── metrics.jsonl                 # one JSON line per epoch
```

Every checkpoint stores `model`, `ema`, `optimizer`, `scheduler`, `scaler`, `epoch`,
`validation`, and the **complete `args`** — which is how `test_native.py` reconstructs
the **exact** architecture without you passing the hyperparameters again. Saving goes
through `atomic_torch_save` (write `.tmp`, then `os.replace`), so a process killed
mid-write never leaves a corrupt checkpoint.

---

## 7. Installation

```bash
# 1) Install PyTorch first, following the official instructions for your CUDA build:
#    https://pytorch.org/get-started/locally/
#    (On Kaggle/Colab: do NOT reinstall torch/torchvision — it breaks CUDA support)

# 2) Everything else
pip install -r requirements.txt      # numpy, Pillow, tqdm
```

Run inside a GPU container:

```bash
./run-gpu-container.sh
docker exec -it road-detection-gpu bash
```

---

## 8. Usage

### 8.1. Training — Massachusetts (single GPU)

```bash
python train.py \
  --dataset massachusetts \
  --data_root /path/to/massachusets \
  --epochs 200 --crop_size 1024 \
  --batch_size 2 --accumulation_steps 2 \
  --lr 2e-4 --bilateral_fusion spatial \
  --road_occlusion_probability 0.3 \
  --save_dir ./checkpoints/mass_dualbranch
```

### 8.2. Training — DeepGlobe with the overlapping protocol

```bash
python train.py \
  --dataset deepglobe \
  --data_root /path/to/deepglobe \
  --deepglobe_train_count 5000 \
  --deepglobe_val_from_test_count 300 \
  --epochs 120 --crop_size 1024 \
  --save_dir ./checkpoints/dg_dualbranch
```

### 8.3. Multi-GPU DDP

```bash
torchrun --nproc_per_node=2 train.py --dataset massachusetts --data_root /path/...
```

### 8.4. Transfer from a trained checkpoint with progressive unfreezing

```bash
python train.py --dataset deepglobe \
  --pretrained_checkpoint ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --transfer_weights ema
# --progressive_unfreeze turns on automatically;
# only the newly introduced spatial-gate tensors are allowed to be missing
```

### 8.5. Exact resumption

```bash
python train.py --dataset massachusetts --resume ./checkpoints/mass_dualbranch/last.pt
```

`--resume` restores the **full** model / EMA / optimizer / scheduler / scaler / epoch /
best-score state. It cannot be combined with `--pretrained_checkpoint`.

### 8.6. Final evaluation — `test_native.py`

```bash
# Massachusetts: calibrate the threshold on val61 (the only place it is permitted)
python test_native.py --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset val61 --search-threshold

# then apply the chosen threshold to test117
python test_native.py --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset test117 --thr 0.46 --tta-mode flip4 --out cache_test117.npz

# DeepGlobe (reads split_manifest.json next to the checkpoint)
python test_native.py --ckpt ./checkpoints/dg_dualbranch/best_fixed_road_iou.pt \
  --subset deepglobe_test1226
```

Key options:

| Flag | Meaning |
|---|---|
| `--weights ema\|model` | defaults to `ema` |
| `--deploy` | fuse the Rep-blocks **after** loading the checkpoint (the `.npz` cache is renamed automatically so caches never mix) |
| `--tta-mode none\|roadx3\|flip4\|d4` | `none` exactly matches training-time validation |
| `--tta-merge probabilities\|logits` | `probabilities` recommended |
| `--out cache.npz` | save/reload float32 probability maps and labels (change the threshold without re-running the model) |
| `--search-threshold` | **allowed only on `val61` / `deepglobe_val300`** — prevents test-set leakage |

> **`roadx3`** is a compatibility profile for the supplied `roadx.infer` code: pad to a
> stride multiple, use three views (identity / horizontal / vertical flip), blend
> per-tile probabilities **uniformly**, and **inverse-transform the complete canvas** —
> which fixes the coordinate bug in the original (inverse-flipping each tile while it
> still sits at the transformed tile coordinate).

### 8.7. Re-parameterization verification — `compare_reparameterization.py`

```bash
python compare_reparameterization.py \
  --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset val61 --json-out rep_report.json
```

In a **single run**, the script:

1. builds both the multi-branch and the deploy form from the **same** checkpoint;
2. checks FP32 equivalence (`--atol` / `--rtol`, default 1e-4) and **raises rather than
   allowing deployment** if the tolerance is exceeded;
3. measures GMACs/GFLOPs (hooks on Conv/Linear), latency, throughput and peak VRAM;
4. evaluates **both forms** on the same images, threshold and inference path, and
   prints the deltas.

Trim the run with `--skip-benchmark` / `--skip-eval`.

---

## 9. Source Map

| File | Contents |
|---|---|
| [modeling/model.py](modeling/model.py) | `TruncatedResNet34`, `ProgressiveDAPPM`, `ResidualSpatialGate`, `ControlledRoadFusion`, `DualResolutionContext`, `DualBranchRoadNet`, `build_model` |
| [modeling/decoder.py](modeling/decoder.py) | `ConvBNAct` / `ConvGNAct`, `RepVGGBlock`, `RepDepthwiseBlock`, `RoadReconstructionDecoder`, `soft_skeletonize`, `RoadSegCenterlineTverskyLoss`, `verify_reparameterization` |
| [train.py](train.py) | DDP, dataset discovery, split resolution, dataset/augmentation, optimizer/scheduler/EMA, training loop, sliding-window validation, checkpointing |
| [test_native.py](test_native.py) | Native-resolution evaluation with TTA, threshold calibration, `.npz` caching |
| [compare_reparameterization.py](compare_reparameterization.py) | Equivalence + FLOPs/latency/VRAM + evaluation of both model forms |
| [run-gpu-container.sh](run-gpu-container.sh) | CUDA PyTorch container mounting the project at `/workspace` |

---

## 10. Summary of Technical Contributions

1. **A persistent S8 detail stream** plus **exactly one** S8↔S16 bilateral exchange —
   avoiding the cost of two bilateral modules and avoiding any demand that S32 preserve
   thin roads.
2. **Zero-initialized `ResidualSpatialGate`** — a single-channel spatial gate that
   starts at exactly 1.0, making it a fully **backward-compatible extension** of the
   static residual exchange.
3. **`ProgressiveDAPPM` with GroupNorm** — progressive multi-scale context that stays
   valid for small batches and for the 1×1 global pooling branch.
4. **Centerline Tversky supervision ($\beta > \alpha$)** with a pure-PyTorch
   morphological skeleton, targets generated **before** downsampling, and a ramp-up
   schedule.
5. **Road-guided occlusion augmentation** — teaches the model to infer occluded road
   segments without polluting the labels.
6. **Verified re-parameterization** — a DW 5×5 kernel fused from 3×3 / 1×5 / 5×1 /
   identity, with a script that proves equivalence and quantifies the deployment gain.
7. **A strict evaluation protocol** — native resolution, Hann logit blending, splits
   written to a manifest, threshold calibration **restricted** to validation, and
   relaxed F1 for topology.
