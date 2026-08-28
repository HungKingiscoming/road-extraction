# DualBranchRoadNet — Road Extraction from Remote Sensing Imagery

Research code for **road extraction** (binary road segmentation) from
high-resolution satellite and aerial imagery, evaluated on two standard
benchmarks: **Massachusetts Roads** and **DeepGlobe Road Extraction**.

The entire pipeline — data handling, model, loss, training, inference and
profiling — is written in plain PyTorch with no dependency on
`mmsegmentation`, `segmentation_models_pytorch` or `albumentations`. It runs on a
single GPU or on multiple GPUs via DDP (tested on Kaggle T4×2).

---

## 1. Problem Statement

### 1.1. Formal definition

Given an optical RGB image $I \in \mathbb{R}^{H \times W \times 3}$ captured from a
satellite or aircraft, learn the mapping

$$f_\theta : I \longmapsto \hat{Y} \in \{0, 1\}^{H \times W}$$

where $\hat{Y}_{ij} = 1$ if pixel $(i,j)$ belongs to a **road surface** and $0$
otherwise (buildings, trees, farmland, rivers, parking lots, …).

The model emits **two-channel logits** at the full input resolution
($\hat{L} \in \mathbb{R}^{2 \times H \times W}$). The predicted mask is obtained by
thresholding the road-class probability $p = \mathrm{softmax}(\hat{L})_1 \ge \tau$
(default $\tau = 0.5$, with an optional threshold-calibration mode fitted on the
validation split).

This is **binary semantic segmentation**, but several properties make it behave
very differently from ordinary object segmentation (Cityscapes, ADE20K, …).

### 1.2. Why this problem is hard

| # | Challenge | Technical consequence |
|---|---|---|
| 1 | **Extreme class imbalance.** Roads cover only ~2–5% of the pixels. | Plain cross-entropy collapses towards background; class weighting plus Dice is required. Accuracy is meaningless — evaluation must use IoU/F1 of the **road class alone**. |
| 2 | **Thin, elongated structures.** Roads are 8–20 px wide at 1 m/px. | At stride 32 (S32) a road is well under one pixel and simply disappears. A standard encoder–decoder that downsamples deeply and then interpolates back cannot recover it. |
| 3 | **Topology matters more than pixels.** Trees, shadows, overpasses and vehicles occlude road segments. | A 5 px break destroys an entire route while barely moving IoU. This calls for **centerline** supervision and a **relaxed F1** metric. |
| 4 | **Long-range context is required.** Rivers, field boundaries, fences and long rooftops all look like linear features. | A semantic branch with a large receptive field (multi-scale pooling) is needed to disambiguate them from regional context. |
| 5 | **Training/inference resolution mismatch.** Training uses 1024 crops; native Massachusetts images are 1500×1500. | Evaluation must run at **native resolution** with overlapping sliding windows, blending logits through a Hann window to remove tile seams. |
| 6 | **Deployment cost.** Parallel branches slow inference down. | **Re-parameterizable** blocks: train multi-branch, then fuse exactly into a single convolution for deployment. |

### 1.3. Out of scope

- No **road graph** extraction (nodes/edges, vectorization) — raster masks only.
- No **road type** classification (highway / urban / dirt).
- No multispectral, SAR, or time-series input.

---

## 2. Data and Split Protocols

### 2.1. Massachusetts Roads

- 1500×1500 px images at **1.0 m/pixel** resolution.
- Directory layout: `images/`, `labels/`, `train.txt`, `test.txt`.
- Splits follow the **exact order given in the txt files** (`pairs_from_list`), never
  a random shuffle:
  - `train.txt` → training set.
  - `test.txt`: the **first 61 entries become validation**, the remaining **117
    images** form the final test set.
  - Controlled by `--test_eval_images` (default 61).
- The trainer **cross-checks** that train and test do not intersect and raises an
  error on any overlapping key.

### 2.2. DeepGlobe Road Extraction

- 1024×1024 px images at **0.5 m/pixel**, 6226 publicly labeled image–mask pairs.
- The `valid/` directory of many mirrors ships **without masks**, so
  `_first_labeled_pair` accepts a directory only if masks actually pair up;
  otherwise it falls back to a deterministic holdout carved out of the training set.
- **Paper-compatible "overlapping" protocol** (enabled by `--deepglobe_train_count`):
  1. Deterministic shuffle seeded by `--split_seed` (default 3407).
  2. The first `train_count` pairs become the training set (e.g. 5000).
  3. **Every** remaining pair forms the test set (1226 images).
  4. Validation is the first `--deepglobe_val_from_test_count` pairs of that test set
     (300 images).

  → **val300 ⊂ test1226 is intentional**, and is recorded explicitly in
  `split_manifest.json` under `overlap_counts.val_test`.

### 2.3. `split_manifest.json` — reproducibility

On every run, rank 0 writes `save_dir/split_manifest.json` containing:

- `split_seed` and `counts` (train/val/test),
- `overlap_counts` (train↔val, train↔test, val↔test),
- the **full absolute paths** of every pair in all three splits.

For DeepGlobe, `test_native.py` reads that same file back (`pairs_from_manifest`), so
the evaluation set is **identical** to the one used during training — random indices
are never regenerated.

### 2.4. Mask loading

`read_binary_mask` handles both $\{0,1\}$ and $\{0,255\}$ label encodings: the
threshold is 0 when `max(mask) <= 1`, and 127 otherwise. Three-channel label images
are reduced with a per-pixel channel `max`.

---

## 3. Architecture — `DualBranchRoadNet`

The core idea: **keep one detail stream permanently at S8** (never downsample it
deeply) and let a separate **semantic stream** descend to S16/S32 to gather context,
then inject that context back into the detail stream **under explicit control**
through gated residuals.

```
                          ResNet-34 (pretrained, shared)
 image ─► stem S2(64) ─► layer1 S4(64) ─► layer2 S8(128) ─► layer3 S16(256) ─► layer4 S32(512)
             │               │                │                  │                    │
             │               │                ▼                  │                    ▼
             │               │       detail_proj 1×1 → 96ch      │           semantic_proj 1×1 → 192ch
             │               │                │                  │                    │
             │               │        [RepVGG ×2] ◄─ bilateral exchange ─► ProgressiveDAPPM
             │               │                │      S8 ↔ S16 (spatial gate)   (pool 8,4,2,1 + GN)
             │               │        [RepVGG ×2]                │                    │
             │               │                │                  ◄─── gated residual ─┘
             │               │                ▼                  │
             │               │     ControlledRoadFusion(96) ◄── semantic_to_fusion 1×1 ◄┘
             │               │                │
             ▼               ▼                ▼
       stem_proj(16)   shallow_proj(32)   fused S8 (96)
             │               │                │
             │               └──► S4: concat → fuse 64ch → RepDW ×2
             └─────────────────► S2: concat → fuse 32ch → RepDW ×2
                                       │           └─► centerline head (TRAIN ONLY, at S4)
                                       ▼
                        upsample → SeparableConv 24ch → dropout → conv 1×1 → logits 2×H×W
```

### 3.1. Encoder — `TruncatedResNet34`

A torchvision ResNet-34 returning **five feature levels**: `stem_s2(64)`,
`shallow_s4(64)`, `shared_s8(128)`, `semantic_s16(256)`, `semantic_s32(512)`.

Weights come from ImageNet, or from a local `.pth` file via
`--encoder_weights_path` (useful on machines without internet access).
`_extract_state_dict` strips the `module.` / `encoder.backbone.` / `backbone.`
prefixes automatically and **raises if fewer than 100 tensors match**, guarding
against loading the wrong checkpoint.

### 3.2. `DualResolutionContext` — two-resolution streams

**Detail stream (S8, 96 channels).** `layer2` (128ch) → 1×1 → 96ch, through two
`RepVGGBlock`s, then it receives semantic evidence, then two more `RepVGGBlock`s.
This stream **never leaves S8**, so thin road evidence is never destroyed.

**Semantic stream.** Anchored at `layer3` (S16, 256ch — pretrained features).
`layer4` (S32, 512ch) → 1×1 → 192ch is used **only to gather broad context**.

**There is exactly one genuine bilateral exchange: S8 ↔ S16.** This is a deliberate
design choice — it avoids asking the S32 map to preserve thin roads, and avoids a
second, expensive bilateral module.

Both directions are **residual with learnable, conservatively initialized scales**:

| Path | Initialization | Rationale |
|---|---|---|
| `semantic_to_detail_scale_1` | `0.10` | semantics **weakly assist** detail |
| `detail_to_semantic_scale_1` | `0.0` | the randomly initialized detail branch **must not disturb** pretrained semantics immediately |
| `context_scale` (S32→S16) | `0.10` | DAPPM context returns to S16 |
| `fusion_scale` (final fusion) | `0.10` | semantics enter the detail stream at the fusion step |

**`ResidualSpatialGate` (with `--bilateral_fusion spatial`).** It predicts **one**
single-channel spatial modulation map from GroupNorm-normalized target and source
features:

$$g = 2\,\sigma\!\left(\mathrm{Conv}(\cdot)\right), \qquad \text{last conv zero-initialized} \;\Rightarrow\; g \equiv 1$$

Because of the zero-init, the model **starts out numerically identical** to the
`static` variant (channel-scaled residuals), and only gradually learns to suppress
clutter and strengthen road-shaped regions — no abrupt shift in the pretrained
feature distribution. Using a **single-channel** gate rather than a C-channel
attention map is intentional: road/background selection is primarily **spatial**,
while channel selectivity is already handled by the learned residual scales. The
result is fewer parameters and less risk of overfitting Massachusetts.

The gate's `last_mean` / `last_std` are logged every epoch (`gate_statistics()`) so
you can verify that each information route is actually being used.

### 3.3. `ProgressiveDAPPM` — progressive multi-scale context

At S32, adaptive pooling runs over grids `(8, 4, 2, 1)` — **finest to coarsest**.
Each branch is projected with a 1×1 convolution, interpolated back to the original
size, **added to the previous stage's representation**, and only then processed by a
3×3 convolution. Each stage therefore adds broader context *progressively*, unlike
the parallel additions of ASPP/PPM. All stages are finally concatenated, compressed
with a 1×1 convolution, and added to a shortcut.

It uses **GroupNorm rather than BatchNorm**: the global pooling branch produces a 1×1
map, where BatchNorm would be meaningless and unstable with small per-GPU batches
(default `batch_size=2`).

### 3.4. `ControlledRoadFusion` — final fusion at S8

The two streams are **normalized independently** and then **concatenated** (not
summed element-wise, which would destroy channel identities) → 1×1 → multiplied by a
learnable per-channel scale (init 0.10) → added residually to the detail stream →
refined by a directional `RepDepthwiseBlock`.

### 3.5. `RoadReconstructionDecoder` — S8 → S1

- S8(96) → 1×1 → 64 → upsample to S4 → concat `shallow_proj(64→32)` → fuse to 64 → **RepDW ×2**
- → upsample to S2 → concat `stem_proj(64→16)` → fuse to 32 → **RepDW ×2**
- → upsample to the exact `output_size` → `SeparableConvBNAct(32→24)` → dropout 0.05 → 1×1 conv → **2 channels**
- An **auxiliary centerline head** at S4 (`ConvBNAct(64→32)` + 1×1 conv → 1 channel)
  is active **only when `model.training == True`**. At eval time `forward` returns a
  single logits tensor, so it costs nothing at inference.

### 3.6. Re-parameterization (multi-branch training → single-branch deployment)

| Block | Training form | Deployment form |
|---|---|---|
| `RepVGGBlock` | Conv3×3-BN + Conv1×1-BN + BN identity | **one 3×3 conv with bias** |
| `RepDepthwiseBlock` | DW3×3-BN + DW1×5-BN + DW5×1-BN + BN identity (+ pointwise) | **one DW 5×5 conv with bias** (+ unchanged pointwise) |

The fusion is **mathematically exact** (fold BN into the convolution, pad kernels to
5×5, sum them). The 1×5 / 5×1 branches learn **directional** road geometry
separately, which matches the horizontally and vertically elongated structure of
roads.

Call `model.switch_to_deploy()`. Verify the error with
`verify_reparameterization()` or with the `compare_reparameterization.py` script.

### 3.7. Parameter counts (default configuration)

| Component | Parameters |
|---|---|
| ResNet-34 encoder | 21,284,672 |
| `dual_branch` | 1,203,282 |
| `decode_head` | 53,891 |
| **Total (training form)** | **22,541,845** |
| **Total (after `switch_to_deploy`)** | **22,502,773** |

> Everything "new" relative to the standard backbone accounts for only ~1.26M
> parameters (≈5.6% of the total).

---

## 4. Loss — `RoadSegCenterlineTverskyLoss`

$$\mathcal{L} = \underbrace{\mathrm{CE}_w}_{\text{class balance}} \;+\; \lambda_{\text{dice}} \underbrace{\mathcal{L}_{\text{Dice}}}_{\text{region}} \;+\; \lambda_{\text{aux}}(t) \underbrace{\mathcal{L}^{\text{centerline}}_{\text{Tversky}}}_{\text{connectivity}}$$

**(a) Weighted cross-entropy.** The road-class weight is computed automatically at
startup:

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
