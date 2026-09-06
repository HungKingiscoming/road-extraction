from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    from torchvision.ops import DeformConv2d
except ImportError:  # pragma: no cover
    DeformConv2d = None


class ConvBNAct(nn.Sequential):
    """Convolution, BatchNorm, and an optional ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True,
        zero_init_bn: bool = False,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        convolution = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        norm = nn.BatchNorm2d(out_channels)
        if zero_init_bn:
            nn.init.zeros_(norm.weight)
        layers: list[nn.Module] = [convolution, norm]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvGNAct(nn.Sequential):
    """Conv-GroupNorm-ReLU used on pooled maps, including 1x1 maps."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: Optional[int] = None,
        activation: bool = True,
    ) -> None:
        if padding is None:
            padding = kernel_size // 2
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(groups, out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class DeformableConvBlock(nn.Module):
    """Deformable, dense-conv drop-in thay cho RepVGGBlock.

    Học một trường offset (+ mask điều biến) trực tiếp từ input để lấy mẫu
    kernel tại vị trí không cố định trên lưới vuông -- biểu diễn được đường
    cong/chéo ở mọi hướng, không chỉ ngang/dọc/45 độ cố định như conv thường.

    Khác biệt so với RepVGGBlock: offset phụ thuộc input nên KHÔNG fuse được
    thành một conv tĩnh. deploy=True raise lỗi rõ ràng thay vì âm thầm sai.
    switch_to_deploy() là no-op chỉ để tương thích với các vòng quét
    isinstance hiện có.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        deploy: bool = False,
        kernel_size: int = 3,
        offset_kernel: int = 3,
    ) -> None:
        super().__init__()
        if DeformConv2d is None:
            raise ImportError(
                "DeformableConvBlock cần torchvision.ops.DeformConv2d "
                "(torchvision >= 0.8)."
            )
        if deploy:
            raise ValueError(
                "DeformableConvBlock không hỗ trợ deploy=True: offset tính "
                "từ input nên không thể fuse thành một conv tĩnh duy nhất."
            )
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        padding = self.kernel_size // 2
        offset_padding = int(offset_kernel) // 2
        taps = self.kernel_size * self.kernel_size

        # offset_groups=1 (suy ra từ số kênh offset_conv sinh ra, độc lập với
        # groups của deform_conv): một trường offset dùng chung cho mọi
        # channel -- vừa rẻ vừa hợp lý vì hướng đường cục bộ là thuộc tính
        # không gian, không phải thuộc tính riêng của từng channel.
        self.offset_conv = nn.Conv2d(
            self.in_channels, 2 * taps, offset_kernel,
            stride=self.stride, padding=offset_padding,
        )
        self.mask_conv = nn.Conv2d(
            self.in_channels, taps, offset_kernel,
            stride=self.stride, padding=offset_padding,
        )
        # Zero-init: lúc bắt đầu train, offset=0 (lấy mẫu đúng lưới chuẩn,
        # hệt conv thường) và mask ~ 1 (sigmoid(4)~=0.98) -- không phá vỡ
        # feature pretrained ngay từ epoch đầu.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 4.0)

        self.deform_conv = DeformConv2d(
            self.in_channels, self.out_channels, self.kernel_size,
            stride=self.stride, padding=padding, bias=False,
        )
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.use_residual = (
            self.in_channels == self.out_channels and self.stride == 1
        )

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        out = self.norm(self.deform_conv(x, offset, mask))
        if self.use_residual:
            out = out + x
        return self.activation(out)

    def switch_to_deploy(self) -> None:
        return  # no-op, xem docstring


class DeformableRoadRefineBlock(nn.Module):
    """Drop-in thay cho RepDepthwiseBlock: depthwise deformable 5x5 tinh
    chỉnh road geometry ở decoder (final fusion, S4, S2).

    offset_conv/mask_conv chỉ sinh MỘT trường offset dùng chung cho mọi
    channel (offset_groups=1, độc lập với groups=channels của deform_conv --
    torchvision suy ra offset_groups từ số kênh của chính tensor offset,
    tách biệt hoàn toàn với groups của weight). Biến dạng hình học tại một
    vị trí không gian là thuộc tính chia sẻ giữa các channel của cùng một
    pixel, không cần offset riêng cho từng channel -- giữ offset predictor rẻ.

    Không hỗ trợ deploy=True: offset phụ thuộc input nên không fuse được
    thành một conv tĩnh duy nhất như RepDepthwiseBlock gốc.
    """

    def __init__(
        self,
        channels: int,
        deploy: bool = False,
        kernel_size: int = 5,
        offset_kernel: int = 3,
    ) -> None:
        super().__init__()
        if DeformConv2d is None:
            raise ImportError(
                "DeformableRoadRefineBlock cần torchvision.ops.DeformConv2d "
                "(torchvision >= 0.8)."
            )
        if deploy:
            raise ValueError(
                "DeformableRoadRefineBlock không hỗ trợ deploy=True: offset "
                "phụ thuộc input nên không thể fuse thành một conv tĩnh."
            )
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        padding = self.kernel_size // 2
        offset_padding = int(offset_kernel) // 2
        taps = self.kernel_size * self.kernel_size

        self.offset_conv = nn.Conv2d(
            self.channels, 2 * taps, offset_kernel, padding=offset_padding
        )
        self.mask_conv = nn.Conv2d(
            self.channels, taps, offset_kernel, padding=offset_padding
        )
        # Zero-init: offset=0 lúc bắt đầu train (lấy mẫu đúng lưới chuẩn, hệt
        # depthwise conv thường), mask ~ 1 (sigmoid(4)~=0.98) -- không phá vỡ
        # hành vi đã học của checkpoint transfer ngay từ epoch đầu.
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.constant_(self.mask_conv.bias, 4.0)

        self.deform_conv = DeformConv2d(
            self.channels, self.channels, self.kernel_size,
            padding=padding, groups=self.channels, bias=False,
        )
        self.norm = nn.BatchNorm2d(self.channels)
        self.spatial_activation = nn.ReLU(inplace=True)
        self.pointwise = ConvBNAct(
            self.channels, self.channels, 1, padding=0, activation=False
        )
        self.output_activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mask_conv(x))
        spatial = self.spatial_activation(self.norm(self.deform_conv(x, offset, mask)))
        return self.output_activation(x + self.pointwise(spatial))

    def switch_to_deploy(self) -> None:
        return  # no-op, xem docstring


class SeparableConvBNAct(nn.Sequential):
    """Depthwise-separable full-resolution prediction refinement."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            ConvBNAct(
                in_channels,
                in_channels,
                3,
                groups=in_channels,
            ),
            ConvBNAct(in_channels, out_channels, 1, padding=0),
        )


class RoadReconstructionDecoder(nn.Module):
    """S8-to-S1 decoder with one train-only S4 centerline head."""

    def __init__(
        self,
        stem_channels: int = 64,
        shallow_channels: int = 64,
        fused_channels: int = 96,
        s4_channels: int = 64,
        s2_channels: int = 32,
        full_channels: int = 24,
        num_classes: int = 2,
        dropout: float = 0.05,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        shallow_skip_channels = max(24, s4_channels // 2)
        stem_skip_channels = max(16, s2_channels // 2)

        self.fused_proj = ConvBNAct(fused_channels, s4_channels, 1, padding=0)
        self.shallow_proj = ConvBNAct(
            shallow_channels, shallow_skip_channels, 1, padding=0
        )
        self.s4_fuse = ConvBNAct(
            s4_channels + shallow_skip_channels,
            s4_channels,
            1,
            padding=0,
        )
        self.s4_refine = nn.Sequential(
            DeformableRoadRefineBlock(s4_channels, deploy=deploy),
            DeformableRoadRefineBlock(s4_channels, deploy=deploy),
        )

        self.stem_proj = ConvBNAct(
            stem_channels, stem_skip_channels, 1, padding=0
        )
        self.s2_fuse = ConvBNAct(
            s4_channels + stem_skip_channels,
            s2_channels,
            1,
            padding=0,
        )
        self.s2_refine = nn.Sequential(
            DeformableRoadRefineBlock(s2_channels, deploy=deploy),
            DeformableRoadRefineBlock(s2_channels, deploy=deploy),
        )

        self.full_refine = SeparableConvBNAct(s2_channels, full_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(full_channels, num_classes, 1)

        auxiliary_channels = max(24, s4_channels // 2)
        self.centerline_head = nn.Sequential(
            ConvBNAct(s4_channels, auxiliary_channels, 3),
            nn.Conv2d(auxiliary_channels, 1, 1),
        )

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        stem_s2: Tensor,
        shallow_s4: Tensor,
        fused_s8: Tensor,
        output_size: Tuple[int, int],
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        p4 = self._resize(self.fused_proj(fused_s8), shallow_s4.shape[-2:])
        p4 = self.s4_fuse(torch.cat((p4, self.shallow_proj(shallow_s4)), dim=1))
        p4 = self.s4_refine(p4)

        p2 = self._resize(p4, stem_s2.shape[-2:])
        p2 = self.s2_fuse(torch.cat((p2, self.stem_proj(stem_s2)), dim=1))
        p2 = self.s2_refine(p2)

        full = self._resize(p2, output_size)
        road_logits = self.classifier(self.dropout(self.full_refine(full)))
        if self.training:
            return self.centerline_head(p4), road_logits
        return road_logits

    def switch_to_deploy(self) -> None:
        """Không còn tác dụng: toàn bộ block refine giờ là deformable,
        offset phụ thuộc input nên không có dạng conv tĩnh để fuse về."""
        return


def _soft_erode(mask: Tensor) -> Tensor:
    vertical = -F.max_pool2d(-mask, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-mask, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(vertical, horizontal)


def _soft_dilate(mask: Tensor) -> Tensor:
    return F.max_pool2d(mask, 3, stride=1, padding=1)


def _soft_open(mask: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(mask))


def soft_skeletonize(mask: Tensor, iterations: int = 8) -> Tensor:
    """Morphological skeleton target generation using only PyTorch ops."""
    opened = _soft_open(mask)
    skeleton = F.relu(mask - opened)
    for _ in range(max(0, int(iterations))):
        mask = _soft_erode(mask)
        opened = _soft_open(mask)
        delta = F.relu(mask - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton.clamp_(0.0, 1.0)


def binary_dice_loss(
    probability: Tensor, target: Tensor, eps: float = 1e-6
) -> Tensor:
    probability = probability.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def binary_tversky_loss(
    probability: Tensor,
    target: Tensor,
    alpha: float = 0.30,
    beta: float = 0.70,
    eps: float = 1e-6,
) -> Tensor:
    """One centerline loss; beta > alpha penalizes broken roads more."""
    probability = probability.float().flatten(1)
    target = target.float().flatten(1)
    true_positive = (probability * target).sum(dim=1)
    false_positive = (probability * (1.0 - target)).sum(dim=1)
    false_negative = ((1.0 - probability) * target).sum(dim=1)
    score = (true_positive + eps) / (
        true_positive
        + float(alpha) * false_positive
        + float(beta) * false_negative
        + eps
    )
    return (1.0 - score).mean()


class RoadSegCenterlineTverskyLoss(nn.Module):
    """Compact road objective: weighted CE + Dice + centerline Tversky."""

    def __init__(
        self,
        road_class_weight: float = 2.0,
        main_dice_weight: float = 1.0,
        aux_weight: float = 0.15,
        centerline_alpha: float = 0.30,
        centerline_beta: float = 0.70,
        skeleton_iterations: int = 8,
        centerline_dilation: int = 1,
        fast_centerline_target: bool = False,
    ) -> None:
        super().__init__()
        self.road_class_weight = float(road_class_weight)
        self.main_dice_weight = float(main_dice_weight)
        self.aux_weight = float(aux_weight)
        self.centerline_alpha = float(centerline_alpha)
        self.centerline_beta = float(centerline_beta)
        self.skeleton_iterations = int(skeleton_iterations)
        self.centerline_dilation = int(centerline_dilation)
        self.fast_centerline_target = bool(fast_centerline_target)

    def forward(
        self,
        outputs: Tuple[Tensor, Tensor],
        target: Tensor,
    ) -> Dict[str, Tensor]:
        centerline_logits, road_logits = outputs
        labels = (target > 0).long()
        road_mask = labels.unsqueeze(1).float()
        class_weights = road_logits.new_tensor([1.0, self.road_class_weight])
        loss_main_ce = F.cross_entropy(
            road_logits.float(), labels, weight=class_weights
        )
        road_probability = road_logits.float().softmax(dim=1)[:, 1:2]
        loss_main_dice = binary_dice_loss(road_probability, road_mask)

        with torch.no_grad():
            # Skeletonize before reducing resolution.  This preserves narrow
            # branches and intersections that can merge when the mask is
            # max-pooled directly to S4.
            if self.fast_centerline_target:
                intermediate_size = tuple(
                    min(source, target_size * 2)
                    for source, target_size in zip(
                        road_mask.shape[-2:], centerline_logits.shape[-2:]
                    )
                )
                skeleton_input = F.adaptive_max_pool2d(
                    road_mask.float(), intermediate_size
                )
                scale = max(
                    road_mask.shape[-2] / max(intermediate_size[0], 1),
                    road_mask.shape[-1] / max(intermediate_size[1], 1),
                )
                target_iterations = (
                    max(1, math.ceil(self.skeleton_iterations / scale))
                    if self.skeleton_iterations > 0
                    else 0
                )
                target_dilation = max(
                    0, int(math.floor(self.centerline_dilation / scale + 0.5))
                )
            else:
                skeleton_input = road_mask.float()
                target_iterations = self.skeleton_iterations
                target_dilation = self.centerline_dilation

            centerline_target = soft_skeletonize(
                skeleton_input, target_iterations
            )
            if target_dilation > 0:
                kernel = 2 * target_dilation + 1
                centerline_target = F.max_pool2d(
                    centerline_target,
                    kernel,
                    stride=1,
                    padding=target_dilation,
                )
            centerline_target = F.adaptive_max_pool2d(
                centerline_target, centerline_logits.shape[-2:]
            )

        loss_centerline = binary_tversky_loss(
            centerline_logits.float().sigmoid(),
            centerline_target,
            alpha=self.centerline_alpha,
            beta=self.centerline_beta,
        )
        total = (
            loss_main_ce
            + self.main_dice_weight * loss_main_dice
            + self.aux_weight * loss_centerline
        )
        return {
            "loss_total": total,
            "loss_main_ce": loss_main_ce.detach(),
            "loss_main_dice": loss_main_dice.detach(),
            "loss_aux_centerline": loss_centerline.detach(),
            "loss_centerline_tversky": loss_centerline.detach(),
        }
