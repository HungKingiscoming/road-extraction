# DualBranchRoadNet — Trích xuất đường giao thông từ ảnh viễn thám

Mã nguồn nghiên cứu cho bài toán **road extraction** (phân đoạn nhị phân đường
giao thông) trên ảnh vệ tinh / ảnh hàng không độ phân giải cao, đánh giá trên hai
bộ dữ liệu chuẩn **Massachusetts Roads** và **DeepGlobe Road Extraction**.

Toàn bộ pipeline (dữ liệu, mô hình, hàm mất mát, huấn luyện, suy luận, đo hiệu
năng) được viết thuần PyTorch — không phụ thuộc `mmsegmentation` /
`segmentation_models_pytorch` / `albumentations` — chạy được trên 1 GPU hoặc DDP
đa GPU (đã kiểm thử trên Kaggle T4×2).

---

## 1. Phát biểu bài toán

### 1.1. Định nghĩa hình thức

Cho ảnh quang học RGB $I \in \mathbb{R}^{H \times W \times 3}$ chụp từ vệ tinh hoặc
máy bay, cần học ánh xạ

$$f_\theta : I \longmapsto \hat{Y} \in \{0, 1\}^{H \times W}$$

trong đó $\hat{Y}_{ij} = 1$ nếu pixel $(i,j)$ thuộc **bề mặt đường**, và $0$ nếu
thuộc nền (nhà cửa, cây cối, đồng ruộng, sông, bãi đỗ xe, …).

Mô hình xuất **logits 2 kênh** ở đúng độ phân giải đầu vào
($\hat{L} \in \mathbb{R}^{2 \times H \times W}$); nhãn dự đoán lấy theo ngưỡng xác
suất lớp đường $p = \mathrm{softmax}(\hat{L})_1 \ge \tau$ (mặc định $\tau = 0.5$,
kèm chế độ hiệu chỉnh ngưỡng trên tập validation).

Đây là bài toán **semantic segmentation nhị phân**, nhưng có những đặc thù khiến
nó **không** giống phân đoạn vật thể thông thường (Cityscapes, ADE20K…).

### 1.2. Vì sao bài toán này khó

| # | Thách thức | Hệ quả kỹ thuật |
|---|---|---|
| 1 | **Mất cân bằng lớp cực đoan.** Đường chỉ chiếm ~2–5% số pixel. | CE thuần bị lệch về nền; cần trọng số lớp + Dice. Accuracy vô nghĩa, phải dùng IoU/F1 của **riêng lớp đường**. |
| 2 | **Cấu trúc mảnh, kéo dài.** Đường rộng 8–20 px ở 1 m/px. | Ở stride 32 (S32) một con đường chỉ còn dưới 1 px → biến mất. Không thể dùng encoder–decoder chuẩn hạ mẫu sâu rồi nội suy lại. |
| 3 | **Tính liên thông (topology) quan trọng hơn pixel.** Cây, bóng đổ, cầu vượt, xe cộ che khuất từng đoạn đường. | Một đứt gãy 5 px làm hỏng toàn tuyến nhưng gần như không đổi IoU. Cần giám sát **trục tim đường (centerline)** và chỉ số **relaxed F1**. |
| 4 | **Cần ngữ cảnh xa.** Sông, ranh thửa ruộng, hàng rào, mái nhà dài đều là vệt tuyến tính giống đường. | Phải có nhánh ngữ nghĩa với trường tiếp nhận lớn (pooling đa tỉ lệ) để phân biệt bằng ngữ cảnh vùng. |
| 5 | **Xung đột độ phân giải huấn luyện ↔ suy luận.** Huấn luyện trên crop 1024; ảnh Massachusetts gốc là 1500×1500. | Phải đánh giá **native resolution** bằng sliding-window có chồng lấn, trộn logits bằng cửa sổ Hann để tránh vệt nối tile. |
| 6 | **Chi phí triển khai.** Nhánh song song làm chậm suy luận. | Dùng khối **re-parameterizable**: huấn luyện đa nhánh, hợp nhất chính xác thành 1 conv khi triển khai. |

### 1.3. Bài toán này **không** giải quyết

- Không trích xuất **đồ thị đường** (node/edge, vector hoá) — chỉ raster mask.
- Không phân loại **loại đường** (cao tốc / nội đô / đường đất).
- Không xử lý ảnh đa phổ, SAR, hay chuỗi thời gian.

---

## 2. Dữ liệu và giao thức chia tập

### 2.1. Massachusetts Roads

- Ảnh 1500×1500 px, độ phân giải **1.0 m/pixel**.
- Bố cục thư mục: `images/`, `labels/`, `train.txt`, `test.txt`.
- Chia tập theo **đúng thứ tự file txt** (`pairs_from_list`), không random:
  - `train.txt` → tập huấn luyện.
  - `test.txt`: **61 dòng đầu → validation**, phần còn lại (**117 ảnh**) → test cuối cùng.
  - Điều khiển bằng `--test_eval_images` (mặc định 61).
- Trainer **kiểm tra chéo** train/test không giao nhau và báo lỗi nếu trùng key.

### 2.2. DeepGlobe Road Extraction

- Ảnh 1024×1024 px, **0.5 m/pixel**, 6226 cặp ảnh–nhãn có nhãn công khai.
- Thư mục `valid/` của nhiều bản mirror **không có mask** → `_first_labeled_pair`
  chỉ chấp nhận thư mục thực sự ghép được nhãn, nếu không sẽ tự chuyển sang
  holdout tất định lấy từ tập train.
- **Giao thức "overlapping" theo paper** (bật bằng `--deepglobe_train_count`):
  1. Xáo trộn tất định theo `--split_seed` (mặc định 3407).
  2. Lấy `train_count` cặp đầu làm train (ví dụ 5000).
  3. **Toàn bộ** phần còn lại là test (1226 ảnh).
  4. Validation = `--deepglobe_val_from_test_count` cặp đầu của test (300 ảnh).

  → **val300 ⊂ test1226 là cố ý**, và được ghi minh bạch trong
  `split_manifest.json` ở trường `overlap_counts.val_test`.

### 2.3. `split_manifest.json` — khả năng tái lập

Mỗi lần chạy, rank 0 ghi `save_dir/split_manifest.json` gồm:

- `split_seed`, `counts` (train/val/test),
- `overlap_counts` (train↔val, train↔test, val↔test),
- **đường dẫn tuyệt đối đầy đủ** của từng cặp trong cả 3 tập.

`test_native.py` đọc lại chính file này cho DeepGlobe (`pairs_from_manifest`) nên
tập test lúc đánh giá **giống hệt** lúc huấn luyện, không sinh lại chỉ số random.

### 2.4. Đọc nhãn

`read_binary_mask` tự nhận diện cả nhãn $\{0,1\}$ và nhãn $\{0,255\}$: ngưỡng = 0
nếu `max(mask) <= 1`, ngược lại 127. Ảnh nhãn 3 kênh được gộp bằng `max` theo kênh.

---

## 3. Kiến trúc mô hình — `DualBranchRoadNet`

Ý tưởng cốt lõi: **giữ một luồng chi tiết luôn ở S8 (không bao giờ hạ mẫu sâu)** và
để một **luồng ngữ nghĩa** riêng đi xuống S16/S32 thu ngữ cảnh, rồi bơm ngược ngữ
cảnh vào luồng chi tiết **một cách có kiểm soát** bằng residual có cổng.

```
                         ResNet-34 (pretrained, dùng chung)
 ảnh ─► stem S2(64) ─► layer1 S4(64) ─► layer2 S8(128) ─► layer3 S16(256) ─► layer4 S32(512)
           │               │                │                  │                    │
           │               │                ▼                  │                    ▼
           │               │       detail_proj 1×1 → 96ch      │           semantic_proj 1×1 → 192ch
           │               │                │                  │                    │
           │               │        [RepVGG ×2] ◄── trao đổi song hướng ──► ProgressiveDAPPM
           │               │                │      S8 ↔ S16 (spatial gate)   (pool 8,4,2,1 + GN)
           │               │        [RepVGG ×2]                │                    │
           │               │                │                  ◄── residual có cổng ┘
           │               │                ▼                  │
           │               │     ControlledRoadFusion(96) ◄── semantic_to_fusion 1×1 ◄┘
           │               │                │
           ▼               ▼                ▼
      stem_proj(16)   shallow_proj(32)   fused S8 (96)
           │               │                │
           │               └──► S4: concat → fuse 64ch → RepDW ×2
           └─────────────────► S2: concat → fuse 32ch → RepDW ×2
                                      │           └─► centerline head (CHỈ khi train, ở S4)
                                      ▼
                        upsample → SeparableConv 24ch → dropout → conv 1×1 → logits 2×H×W
```

### 3.1. Encoder — `TruncatedResNet34`

ResNet-34 torchvision, trả về **5 mức đặc trưng**: `stem_s2(64)`, `shallow_s4(64)`,
`shared_s8(128)`, `semantic_s16(256)`, `semantic_s32(512)`.

Trọng số ImageNet, hoặc file `.pth` cục bộ qua `--encoder_weights_path` (hữu ích khi
máy huấn luyện không có internet). `_extract_state_dict` tự bóc các tiền tố
`module.` / `encoder.backbone.` / `backbone.` và **báo lỗi nếu khớp dưới 100 tensor**
(chống nạp nhầm checkpoint).

### 3.2. `DualResolutionContext` — hai luồng phân giải

**Luồng chi tiết (detail, S8, 96 kênh).** `layer2` (128ch) → 1×1 → 96ch, qua 2 khối
`RepVGGBlock`, nhận thông tin ngữ nghĩa, rồi qua 2 khối `RepVGGBlock` nữa. Luồng này
**không bao giờ rời S8**, nên nét đường mảnh không bị hủy.

**Luồng ngữ nghĩa (semantic).** Neo tại `layer3` (S16, 256ch — đặc trưng đã pretrain).
`layer4` (S32, 512ch) → 1×1 → 192ch, chỉ dùng để **thu ngữ cảnh rộng**.

**Chỉ có một trao đổi song hướng thật sự: S8 ↔ S16.** Đây là lựa chọn thiết kế có
chủ ý — không bắt bản đồ S32 phải giữ đường mảnh, và tránh một module bilateral thứ
hai rất đắt.

Cả hai chiều đều là **residual có hệ số học được, khởi tạo bảo thủ**:

| Đường truyền | Khởi tạo | Ý nghĩa |
|---|---|---|
| `semantic_to_detail_scale_1` | `0.10` | ngữ nghĩa **hỗ trợ nhẹ** cho chi tiết |
| `detail_to_semantic_scale_1` | `0.0` | nhánh chi tiết (khởi tạo ngẫu nhiên) **không được phá** đặc trưng pretrain ngay |
| `context_scale` (S32→S16) | `0.10` | ngữ cảnh DAPPM quay về S16 |
| `fusion_scale` (trong fusion cuối) | `0.10` | ngữ nghĩa vào luồng chi tiết ở bước hợp nhất |

**`ResidualSpatialGate` (khi `--bilateral_fusion spatial`).** Sinh **một** bản đồ điều
biến không gian 1 kênh từ target và source đã chuẩn hoá GroupNorm:

$$g = 2\,\sigma\!\left(\mathrm{Conv}(\cdot)\right), \qquad \text{conv cuối khởi tạo } 0 \;\Rightarrow\; g \equiv 1$$

Nhờ zero-init, mô hình **khởi đầu giống hệt** biến thể `static` (residual nhân hệ số
kênh), rồi mới dần học triệt tiêu nhiễu và tăng cường vùng hình đường — không gây cú
sốc phân phối cho đặc trưng pretrain. Dùng gate **1 kênh** thay vì C kênh là có chủ
ý: việc chọn đường/nền chủ yếu mang tính **không gian**, còn tính chọn lọc theo kênh
đã do các hệ số residual đảm nhiệm → ít tham số hơn và ít nguy cơ overfit
Massachusetts hơn.

`last_mean` / `last_std` của gate được ghi vào log mỗi epoch (`gate_statistics()`) để
kiểm tra từng tuyến thông tin có thực sự được sử dụng hay không.

### 3.3. `ProgressiveDAPPM` — ngữ cảnh đa tỉ lệ lũy tiến

Tại S32: pooling thích nghi trên các lưới `(8, 4, 2, 1)` (**từ mịn đến thô**). Mỗi
nhánh được chiếu 1×1, nội suy về kích thước gốc, **cộng vào biểu diễn của bước trước**
rồi mới xử lý 3×3 — nên mỗi tầng bổ sung ngữ cảnh rộng hơn *một cách lũy tiến*, khác
với ASPP/PPM cộng song song. Cuối cùng concat toàn bộ → nén 1×1 → cộng shortcut.

Dùng **GroupNorm thay BatchNorm**: nhánh pooling toàn cục cho bản đồ 1×1, BatchNorm sẽ
vô nghĩa / không ổn định với batch nhỏ trên mỗi GPU (mặc định `batch_size=2`).

### 3.4. `ControlledRoadFusion` — hợp nhất cuối tại S8

Hai luồng được **chuẩn hoá độc lập** rồi **concat** (không cộng element-wise, để không
phá danh tính kênh) → 1×1 → nhân hệ số kênh học được (init 0.10) → cộng residual vào
luồng chi tiết → tinh chỉnh bằng `RepDepthwiseBlock` có hướng.

### 3.5. `RoadReconstructionDecoder` — S8 → S1

- S8(96) → 1×1 → 64 → upsample S4 → concat `shallow_proj(64→32)` → fuse 64 → **RepDW ×2**
- → upsample S2 → concat `stem_proj(64→16)` → fuse 32 → **RepDW ×2**
- → upsample về đúng `output_size` → `SeparableConvBNAct(32→24)` → dropout 0.05 → conv 1×1 → **2 kênh**
- **Đầu centerline phụ** tại S4 (`ConvBNAct(64→32)` + conv 1×1 → 1 kênh), **chỉ hoạt
  động khi `model.training == True`**. Khi eval, `forward` trả về đúng một tensor
  logits → không tốn chi phí suy luận.

### 3.6. Re-parameterization (huấn luyện đa nhánh → triển khai một nhánh)

| Khối | Dạng huấn luyện | Dạng triển khai |
|---|---|---|
| `RepVGGBlock` | Conv3×3-BN + Conv1×1-BN + BN identity | **1 conv 3×3 có bias** |
| `RepDepthwiseBlock` | DW3×3-BN + DW1×5-BN + DW5×1-BN + BN identity (+ pointwise) | **1 conv DW 5×5 có bias** (+ pointwise giữ nguyên) |

Hợp nhất là **chính xác về mặt toán học** (fold BN vào conv, pad kernel về 5×5, cộng
kernel). Nhánh 1×5 / 5×1 học riêng hình học **có hướng** của đường — phù hợp với cấu
trúc kéo dài theo phương ngang/dọc.

Gọi `model.switch_to_deploy()`. Kiểm chứng sai số bằng `verify_reparameterization()`
hoặc script `compare_reparameterization.py`.

### 3.7. Số tham số (cấu hình mặc định)

| Thành phần | Tham số |
|---|---|
| Encoder ResNet-34 | 21,284,672 |
| `dual_branch` | 1,203,282 |
| `decode_head` | 53,891 |
| **Tổng (dạng huấn luyện)** | **22,541,845** |
| **Tổng (sau `switch_to_deploy`)** | **22,502,773** |

> Phần "mới" so với backbone chuẩn chỉ ~1.26M tham số (≈5.6% tổng số).

---

## 4. Hàm mất mát — `RoadSegCenterlineTverskyLoss`

$$\mathcal{L} = \underbrace{\mathrm{CE}_w}_{\text{cân bằng lớp}} \;+\; \lambda_{\text{dice}} \underbrace{\mathcal{L}_{\text{Dice}}}_{\text{vùng}} \;+\; \lambda_{\text{aux}}(t) \underbrace{\mathcal{L}^{\text{centerline}}_{\text{Tversky}}}_{\text{liên thông}}$$

**(a) Cross-Entropy có trọng số.** Trọng số lớp đường được tính tự động khi khởi động:

$$\text{imbalance} = \frac{N_{\text{nền}}}{N_{\text{đường}}}, \qquad w_{\text{road}} = \min\!\left(\sqrt{\text{imbalance}},\ \text{cap}\right)$$

với `--road_weight_cap` mặc định 2.0. Rank 0 quét toàn bộ mask train rồi `broadcast`
cho các rank khác. Bỏ qua bước quét bằng `--fixed_road_weight`.

**(b) Dice nhị phân** trên xác suất lớp đường — ổn định hoá gradient khi lớp dương hiếm.

**(c) Tversky trên trục tim đường** — đóng góp chính về topology:

- **Sinh nhãn centerline không cần thư viện ngoài:** `soft_skeletonize` làm mảnh hình
  thái học chỉ bằng `max_pool2d` (erode = `-maxpool(-x)`, dilate = `maxpool(x)`,
  open = dilate∘erode), lặp `--skeleton_iterations` lần (mặc định 8).
- **Skeleton hoá TRƯỚC khi hạ mẫu** — nếu max-pool mask thẳng xuống S4 thì các nhánh
  nhỏ và nút giao sẽ dính vào nhau.
- Nở nhẹ (`--centerline_dilation 1`) rồi `adaptive_max_pool` về S4 khớp đầu phụ.
- Tversky với $\alpha = 0.30 < \beta = 0.70$: **phạt false negative nặng hơn**, tức là
  phạt **đường bị đứt** mạnh hơn đường bị thừa → đúng mục tiêu giữ liên thông.
- **Lịch bật dần:** tắt hoàn toàn trước epoch `--aux_start_epoch` (5), rồi tăng tuyến
  tính trong `--aux_warmup_epochs` (5) lên `--aux_weight` (0.15). Lý do: nhánh
  centerline chỉ có ý nghĩa sau khi mask thô đã tương đối đúng.
- `--fast_centerline_target`: skeleton hoá ở mức trung gian S2 (rẻ hơn), tự động quy
  đổi số vòng lặp và bán kính nở theo tỉ lệ.

---

## 5. Chiến lược huấn luyện

### 5.1. Lấy mẫu crop hướng đường (`random_crop_pair`)

Crop ngẫu nhiên trên ảnh chỉ có 2–5% pixel đường sẽ sinh rất nhiều crop **trống hoàn
toàn**. Giải pháp: với xác suất `--road_crop_probability` (0.60):

1. Hạ mẫu mask 8× bằng max-pool (`coarse_max_mask`) → bản đồ ô có đường.
2. Bốc ngẫu nhiên một ô có đường, dịch ngẫu nhiên để tâm không cố định.
3. Thử tối đa `--road_crop_tries` (8) lần, giữ crop có tỉ lệ đường cao nhất; dừng sớm
   khi đạt `--road_crop_min_fraction` (0.002).

Ảnh nhỏ hơn crop được **pad reflect** cho ảnh và **pad 0** cho mask.

### 5.2. Augmentation

- **Bảo toàn nhãn:** lật ngang/dọc + xoay 90° (nhóm D4) — hợp lệ vì ảnh viễn thám
  không có hướng "trên" ưu tiên.
- **Trắc quang:** brightness/contrast (p=0.60), saturation (p=0.35), Gaussian blur
  (p=0.15), nhiễu Gauss (p=0.15).
- **`road_guided_occlusion` (đặc thù bài toán, `--road_occlusion_probability`).** Vẽ
  các mảng che *ngay trên pixel đường có nhãn*: bóng đổ (giảm sáng 0.30–0.65), tán cây
  (màu xanh), hoặc phương tiện (mảng sáng). **Nhãn phân đoạn KHÔNG đổi** → ép mô hình
  *suy luận* đoạn đường bị che từ tính liên tục xung quanh. Mảng che được giữ nhỏ
  (0.6–1.8% cạnh ngắn) để bài toán không trở nên bất khả thi.

### 5.3. Tối ưu hoá

- **AdamW** (fused nếu có CUDA), `--weight_decay 1e-4` chỉ áp cho tensor `ndim > 1`
  (bias / BN / hệ số gate **không** weight decay).
- **Learning rate phân tầng** (nhân với `--lr`, mặc định 2e-4):

  | Nhóm | Hệ số | Giá trị mặc định |
  |---|---|---|
  | `head` (decoder) | 1.00 | 2.0e-4 |
  | `dual_branch` | `--dual_branch_lr_factor` 0.75 | 1.5e-4 |
  | `layer3`+`layer4`, `layer2` | `--backbone_lr_factor` 0.20 | 4.0e-5 |
  | `stem`+`layer1` | `--early_encoder_lr_factor` 0.10 | 2.0e-5 |

  `build_optimizer` **kiểm tra không tham số nào bị trùng hoặc bị bỏ sót**.
- **Lịch LR:** warmup tuyến tính `--warmup_epochs` (5) từ hệ số 0.10 → cosine annealing
  xuống `--min_lr_ratio` (0.02). Cập nhật **theo optimizer step**, không theo epoch.
- **AMP fp16 + GradScaler**, `channels_last`, `cudnn.benchmark`, TF32. Khi GradScaler
  bỏ qua một update (gradient non-finite), scheduler và EMA **cũng không bước** → giữ
  đồng bộ chính xác.
- **Gradient accumulation** `--accumulation_steps` (2). Batch hiệu dụng =
  `batch_size × world_size × accumulation_steps` (mặc định 2×1×2 = 4).
- **Grad clip** norm 3.0. **EMA** decay 0.999 có ramp
  $d_t = 0.999\,(1 - e^{-t/2000})$ — **EMA là trọng số dùng để validate và test**.

### 5.4. Mở băng lũy tiến (progressive unfreezing)

Tự bật khi có `--pretrained_checkpoint`, tắt khi train từ đầu (ghi đè bằng
`--progressive_unfreeze` / `--no-progressive_unfreeze`).

| Phase | Tên | Epoch mặc định | Module được học |
|---|---|---|---|
| 0 | `head_only` | 0–2 | chỉ decoder |
| 1 | `head_plus_dual_branch` | 3–5 | + `dual_branch` |
| 2 | `plus_resnet_layer3_layer4` | 6–11 | + `layer3`, `layer4` |
| 3 | `plus_resnet_layer2` | 12–… | + `layer2` |
| 4 | `all_trainable` | `--unfreeze_all_epoch` (mặc định **-1** = không bao giờ) | + `stem`, `layer1` |

Phần đóng băng được bọc `torch.no_grad()` **ngay trong `forward`** → tiết kiệm cả bộ
nhớ activation, không chỉ chặn gradient. `enforce_frozen_norm_eval()` giữ BatchNorm của
phần đóng băng ở chế độ `eval` để running statistics không trôi (`--freeze_encoder_bn`
mặc định bật cho **toàn bộ** encoder).

> Optimizer, scheduler và reducer DDP luôn được xây khi **mọi tham số còn trainable**
> (`set_trainable_phase(4)` trước khi build) — nhờ vậy đổi phase giữa chừng không làm
> hỏng param group hay DDP hook.

### 5.5. Huấn luyện phân tán (DDP)

- NCCL, `init_method="env://"`, timeout **15 phút** (rank 0 có thể quét vài nghìn mask
  trước collective đầu tiên).
- `find_unused_parameters` chỉ bật khi dùng progressive unfreezing.
- Backward luôn được gọi trên **mọi rank** — DDP lan truyền gradient non-finite,
  GradScaler bỏ qua update nhất quán trên tất cả rank. Cách này loại bỏ một all-reduce
  chặn và hai lần đồng bộ thiết bị ở **mỗi batch khoẻ mạnh**.
- Metric trong epoch gộp bằng **một** lần copy device→host cho 5 giá trị.
- Validation dùng `DistributedEvalSampler` — chia **chính xác**, không nhân bản ảnh đệm
  như `DistributedSampler` (nếu nhân bản, IoU gộp sẽ sai).
- Trước mỗi lần validate, `ema.module` được broadcast từ rank 0.

---

## 6. Đánh giá

### 6.1. Suy luận native resolution

Không resize ảnh. Sliding window `--val_tile_size` (1024) với `--val_overlap` (256) →
stride 768. Mỗi tile được nhân **cửa sổ Hann 2D** (clamp tối thiểu 0.05) và cộng dồn ở
**miền logits**, cuối cùng chia cho bản đồ chuẩn hoá → xoá hoàn toàn vệt nối tile. Ảnh
nhỏ hơn tile được pad `reflect`.

### 6.2. Bộ chỉ số

Tính trên **confusion gộp toàn tập** (pooled), tiền tố `fixed_` (ngưỡng 0.5) và
`calibrated_` (ngưỡng tối ưu):

- `road_iou` — **chỉ số chính**, IoU của riêng lớp đường.
- `background_iou`, `miou`, `precision`, `recall`, `f1`, `accuracy`.
- `fixed_road_iou_macro` — trung bình IoU **theo từng ảnh** (macro).
- `fixed_relaxed_f1` — **F1 nới lỏng ±3 px** (`--relaxed_buffer_px`): dự đoán được tính
  đúng nếu nằm trong vùng nở 3 px của nhãn và ngược lại. Đây là chỉ số phản ánh **chất
  lượng tuyến / topology**, ít nhạy với lệch 1–2 px của nhãn vẽ tay.

**Hiệu chỉnh ngưỡng.** Tích luỹ histogram 1001 bin của xác suất theo lớp thật, rồi quét
ngưỡng 0.20 → 0.80 bước 0.02 chọn ngưỡng cực đại road IoU — **không cần lưu toàn bộ bản
đồ xác suất**. Kết quả ghi ở `calibrated_threshold`.

### 6.3. Đầu ra một lần chạy

```
save_dir/
├── best_fixed_road_iou.pt        # tốt nhất theo IoU @0.5
├── best_calibrated_road_iou.pt   # tốt nhất theo IoU ở ngưỡng hiệu chỉnh
├── last.pt                       # để resume chính xác
├── split_manifest.json           # tái lập tập chia
└── metrics.jsonl                 # 1 dòng JSON / epoch
```

Mỗi checkpoint chứa `model`, `ema`, `optimizer`, `scheduler`, `scaler`, `epoch`,
`validation`, và **`args` đầy đủ** — nhờ vậy `test_native.py` dựng lại **đúng** kiến
trúc mà không cần truyền lại tham số. Lưu bằng `atomic_torch_save` (ghi `.tmp` rồi
`os.replace`) → không bao giờ có file hỏng khi tiến trình bị kill giữa chừng.

---

## 7. Cài đặt

```bash
# 1) Cài PyTorch trước, theo hướng dẫn chính thức cho CUDA của bạn:
#    https://pytorch.org/get-started/locally/
#    (Trên Kaggle/Colab: KHÔNG cài lại torch/torchvision — sẽ hỏng CUDA)

# 2) Phần còn lại
pip install -r requirements.txt      # numpy, Pillow, tqdm
```

Chạy trong Docker có GPU:

```bash
./run-gpu-container.sh
docker exec -it road-detection-gpu bash
```

---

## 8. Cách sử dụng

### 8.1. Huấn luyện — Massachusetts (1 GPU)

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

### 8.2. Huấn luyện — DeepGlobe theo giao thức overlapping

```bash
python train.py \
  --dataset deepglobe \
  --data_root /path/to/deepglobe \
  --deepglobe_train_count 5000 \
  --deepglobe_val_from_test_count 300 \
  --epochs 120 --crop_size 1024 \
  --save_dir ./checkpoints/dg_dualbranch
```

### 8.3. DDP đa GPU

```bash
torchrun --nproc_per_node=2 train.py --dataset massachusetts --data_root /path/...
```

### 8.4. Transfer từ checkpoint đã train + mở băng lũy tiến

```bash
python train.py --dataset deepglobe \
  --pretrained_checkpoint ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --transfer_weights ema
# --progressive_unfreeze tự bật; chỉ tensor spatial-gate mới được phép thiếu
```

### 8.5. Resume chính xác

```bash
python train.py --dataset massachusetts --resume ./checkpoints/mass_dualbranch/last.pt
```

`--resume` khôi phục **đầy đủ** model / EMA / optimizer / scheduler / scaler / epoch /
best score. Không dùng chung với `--pretrained_checkpoint`.

### 8.6. Đánh giá cuối — `test_native.py`

```bash
# Massachusetts: hiệu chỉnh ngưỡng trên val61 (chỉ được phép ở đây)
python test_native.py --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset val61 --search-threshold

# rồi áp ngưỡng đã chọn lên test117
python test_native.py --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset test117 --thr 0.46 --tta-mode flip4 --out cache_test117.npz

# DeepGlobe (đọc split_manifest.json cạnh checkpoint)
python test_native.py --ckpt ./checkpoints/dg_dualbranch/best_fixed_road_iou.pt \
  --subset deepglobe_test1226
```

Tuỳ chọn quan trọng:

| Cờ | Ý nghĩa |
|---|---|
| `--weights ema\|model` | mặc định `ema` |
| `--deploy` | hợp nhất Rep-block **sau khi** nạp checkpoint (cache `.npz` tự đổi tên để không lẫn) |
| `--tta-mode none\|roadx3\|flip4\|d4` | `none` khớp đúng validation lúc train |
| `--tta-merge probabilities\|logits` | khuyến nghị `probabilities` |
| `--out cache.npz` | lưu / nạp lại bản đồ xác suất float32 + nhãn (đổi ngưỡng không cần chạy lại model) |
| `--search-threshold` | **chỉ cho phép trên `val61` / `deepglobe_val300`** — chặn rò rỉ test |

> **`roadx3`** là profile tương thích với code `roadx.infer` được cung cấp: pad về bội
> của stride, 3 view (gốc / lật ngang / lật dọc), trộn **xác suất đều** theo tile, và
> **nghịch đảo phép biến đổi trên toàn canvas** — sửa lỗi toạ độ của bản gốc (nghịch
> đảo từng tile trong khi tile vẫn nằm ở toạ độ đã biến đổi).

### 8.7. Kiểm chứng re-parameterization — `compare_reparameterization.py`

```bash
python compare_reparameterization.py \
  --ckpt ./checkpoints/mass_dualbranch/best_fixed_road_iou.pt \
  --subset val61 --json-out rep_report.json
```

Trong **một lần chạy**, script:

1. dựng cả dạng đa nhánh và dạng deploy từ **cùng một** checkpoint;
2. kiểm tra tương đương FP32 (`--atol` / `--rtol` mặc định 1e-4) — **raise lỗi nếu vượt
   dung sai**, không cho phép triển khai;
3. đo GMACs/GFLOPs (hook trên Conv/Linear), latency, throughput, peak VRAM;
4. đánh giá **cả hai dạng** trên cùng tập ảnh / ngưỡng / đường suy luận và in delta.

Bỏ bớt bằng `--skip-benchmark` / `--skip-eval`.

---

## 9. Bản đồ mã nguồn

| File | Nội dung |
|---|---|
| [modeling/model.py](modeling/model.py) | `TruncatedResNet34`, `ProgressiveDAPPM`, `ResidualSpatialGate`, `ControlledRoadFusion`, `DualResolutionContext`, `DualBranchRoadNet`, `build_model` |
| [modeling/decoder.py](modeling/decoder.py) | `ConvBNAct` / `ConvGNAct`, `RepVGGBlock`, `RepDepthwiseBlock`, `RoadReconstructionDecoder`, `soft_skeletonize`, `RoadSegCenterlineTverskyLoss`, `verify_reparameterization` |
| [train.py](train.py) | DDP, phát hiện dữ liệu, chia tập, dataset/augmentation, optimizer/scheduler/EMA, vòng huấn luyện, validation sliding-window, checkpoint |
| [test_native.py](test_native.py) | Đánh giá native resolution + TTA, hiệu chỉnh ngưỡng, cache `.npz` |
| [compare_reparameterization.py](compare_reparameterization.py) | Tương đương + FLOPs/latency/VRAM + đánh giá hai dạng mô hình |
| [run-gpu-container.sh](run-gpu-container.sh) | Container PyTorch CUDA, gắn thư mục dự án vào `/workspace` |

---

## 10. Tóm tắt đóng góp kỹ thuật

1. **Luồng chi tiết S8 bền vững** + **một** trao đổi song hướng S8↔S16 duy nhất — tránh
   chi phí của hai module bilateral và tránh bắt S32 phải giữ đường mảnh.
2. **`ResidualSpatialGate` zero-init** — gate không gian 1 kênh khởi đầu bằng đúng 1.0,
   nên là **mở rộng tương thích ngược** hoàn toàn của residual tĩnh.
3. **`ProgressiveDAPPM` dùng GroupNorm** — ngữ cảnh đa tỉ lệ lũy tiến, hợp lệ với batch
   nhỏ và với nhánh pooling toàn cục 1×1.
4. **Giám sát centerline bằng Tversky ($\beta > \alpha$)** với skeleton hình thái thuần
   PyTorch, sinh nhãn **trước** khi hạ mẫu, và bật dần theo lịch.
5. **Augmentation che khuất theo đường** — dạy mô hình suy luận đoạn đường bị che mà
   không làm bẩn nhãn.
6. **Re-parameterization có kiểm chứng** — DW 5×5 gộp từ 3×3 / 1×5 / 5×1 / identity, kèm
   script chứng minh tương đương và đo lợi ích triển khai.
7. **Giao thức đánh giá chặt** — native resolution, trộn Hann trên logits, chia tập ghi
   ra manifest, hiệu chỉnh ngưỡng **chỉ** trên validation, relaxed F1 cho topology.
