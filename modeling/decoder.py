from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


class ConvBN(nn.Sequential):
    """Linear Conv-BN branch used by re-parameterizable blocks."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int,
        padding: Union[int, Tuple[int, int]],
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    @property
    def conv(self) -> nn.Conv2d:
        return self[0]

    @property
    def bn(self) -> nn.BatchNorm2d:
        return self[1]


def _fuse_conv_bn(branch: ConvBN) -> Tuple[Tensor, Tensor]:
    weight = branch.conv.weight
    norm = branch.bn
    std = torch.sqrt(norm.running_var + norm.eps)
    scale = norm.weight / std
    return (
        weight * scale.reshape(-1, 1, 1, 1),
        norm.bias - norm.running_mean * scale,
    )


class RepVGGBlock(nn.Module):
    """RepVGG block exactly fused to one dense 3x3 convolution at deploy."""

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        stride: int = 1,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.stride = int(stride)
        self.deploy = bool(deploy)
        self.activation = nn.ReLU(inplace=True)

        if self.deploy:
            self.reparam = nn.Conv2d(
                self.in_channels,
                self.out_channels,
                3,
                stride=self.stride,
                padding=1,
                bias=True,
            )
        else:
            self.branch_3x3 = ConvBN(
                self.in_channels, self.out_channels, 3, self.stride, 1
            )
            self.branch_1x1 = ConvBN(
                self.in_channels, self.out_channels, 1, self.stride, 0
            )
            if self.in_channels == self.out_channels and self.stride == 1:
                self.branch_identity: Optional[nn.BatchNorm2d] = nn.BatchNorm2d(
                    self.in_channels
                )
            else:
                self.branch_identity = None

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            return self.activation(self.reparam(x))
        identity: Union[Tensor, int]
        identity = self.branch_identity(x) if self.branch_identity else 0
        return self.activation(
            self.branch_3x3(x) + self.branch_1x1(x) + identity
        )

    def _fuse_identity_bn(self) -> Tuple[Union[Tensor, int], Union[Tensor, int]]:
        if self.branch_identity is None:
            return 0, 0
        norm = self.branch_identity
        kernel = norm.weight.new_zeros(
            self.out_channels, self.in_channels, 3, 3
        )
        indices = torch.arange(self.in_channels, device=kernel.device)
        kernel[indices, indices, 1, 1] = 1.0
        std = torch.sqrt(norm.running_var + norm.eps)
        scale = norm.weight / std
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            norm.bias - norm.running_mean * scale,
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.reparam.weight, self.reparam.bias
        kernel_3, bias_3 = _fuse_conv_bn(self.branch_3x3)
        kernel_1, bias_1 = _fuse_conv_bn(self.branch_1x1)
        kernel_id, bias_id = self._fuse_identity_bn()
        kernel = kernel_3 + F.pad(kernel_1, (1, 1, 1, 1)) + kernel_id
        return kernel, bias_3 + bias_1 + bias_id

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            stride=self.stride,
            padding=1,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.reparam = reparam
        del self.branch_3x3
        del self.branch_1x1
        del self.branch_identity
        self.deploy = True


class RepDepthwiseBlock(nn.Module):
    """Road refinement with a deployable depthwise 5x5 spatial kernel.

    During training, 3x3, 1x5, 5x1, and identity paths learn complementary
    road geometry. The four paths are exactly fused into one depthwise 5x5
    convolution for inference. The inexpensive pointwise mixer remains.
    """

    def __init__(self, channels: int, deploy: bool = False) -> None:
        super().__init__()
        self.channels = int(channels)
        self.deploy = bool(deploy)
        self.spatial_activation = nn.ReLU(inplace=True)
        self.output_activation = nn.ReLU(inplace=True)
        if self.deploy:
            self.spatial_reparam = nn.Conv2d(
                channels,
                channels,
                5,
                padding=2,
                groups=channels,
                bias=True,
            )
        else:
            self.branch_3x3 = ConvBN(
                channels, channels, 3, 1, 1, groups=channels
            )
            self.branch_1x5 = ConvBN(
                channels, channels, (1, 5), 1, (0, 2), groups=channels
            )
            self.branch_5x1 = ConvBN(
                channels, channels, (5, 1), 1, (2, 0), groups=channels
            )
            self.branch_identity = nn.BatchNorm2d(channels)
        self.pointwise = ConvBNAct(
            channels, channels, 1, padding=0, activation=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.deploy:
            spatial = self.spatial_reparam(x)
        else:
            spatial = (
                self.branch_3x3(x)
                + self.branch_1x5(x)
                + self.branch_5x1(x)
                + self.branch_identity(x)
            )
        spatial = self.spatial_activation(spatial)
        return self.output_activation(x + self.pointwise(spatial))

    @staticmethod
    def _pad_to_5x5(kernel: Tensor) -> Tensor:
        height, width = kernel.shape[-2:]
        pad_h, pad_w = 5 - height, 5 - width
        return F.pad(
            kernel,
            (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            ),
        )

    def _fuse_identity(self) -> Tuple[Tensor, Tensor]:
        norm = self.branch_identity
        kernel = norm.weight.new_zeros(self.channels, 1, 5, 5)
        kernel[:, 0, 2, 2] = 1.0
        std = torch.sqrt(norm.running_var + norm.eps)
        scale = norm.weight / std
        return (
            kernel * scale.reshape(-1, 1, 1, 1),
            norm.bias - norm.running_mean * scale,
        )

    def get_equivalent_kernel_bias(self) -> Tuple[Tensor, Tensor]:
        if self.deploy:
            return self.spatial_reparam.weight, self.spatial_reparam.bias
        kernels, biases = [], []
        for branch in (self.branch_3x3, self.branch_1x5, self.branch_5x1):
            kernel, bias = _fuse_conv_bn(branch)
            kernels.append(self._pad_to_5x5(kernel))
            biases.append(bias)
        kernel_id, bias_id = self._fuse_identity()
        return sum(kernels, kernel_id), sum(biases, bias_id)

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        reparam = nn.Conv2d(
            self.channels,
            self.channels,
            5,
            padding=2,
            groups=self.channels,
            bias=True,
        ).to(device=kernel.device, dtype=kernel.dtype)
        with torch.no_grad():
            reparam.weight.copy_(kernel)
            reparam.bias.copy_(bias)
        self.spatial_reparam = reparam
        del self.branch_3x3
        del self.branch_1x5
        del self.branch_5x1
        del self.branch_identity
        self.deploy = True


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


def _group_count(channels: int, maximum: int = 8) -> int:
    """Largest small GroupNorm divisor, robust for small per-GPU batches."""
    for groups in range(min(maximum, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class StripPoolingModule(nn.Module):
    """Long-range horizontal/vertical context via two 1-D global poolings.

    Square poolings (as in ProgressiveDAPPM) blend context from a compact
    neighborhood. Roads are the opposite of compact: thin, and often run the
    full width or height of a tile. Pooling the feature map down to a single
    column (H, 1) or a single row (1, W), convolving along that strip, and
    broadcasting back lets every pixel see the full extent of its own row
    and column in one pass -- exactly the two directions a road is likely to
    continue in, and a shape square pooling cannot represent efficiently.
    GroupNorm keeps the branch valid at the small per-GPU batch sizes crop
    training uses, matching ProgressiveDAPPM. The block is residual and the
    fuse projection is not zero-initialized (unlike the gates elsewhere in
    this model) because strip pooling is a fixed, useful prior from the
    first step rather than a correction that should start at zero.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        pooled_channels = max(16, channels // max(1, reduction))
        self.reduce = ConvGNAct(channels, pooled_channels, 1, padding=0)
        self.horizontal_conv = nn.Conv2d(
            pooled_channels, pooled_channels, (3, 1), padding=(1, 0), bias=False
        )
        self.horizontal_norm = nn.GroupNorm(
            _group_count(pooled_channels), pooled_channels
        )
        self.vertical_conv = nn.Conv2d(
            pooled_channels, pooled_channels, (1, 3), padding=(0, 1), bias=False
        )
        self.vertical_norm = nn.GroupNorm(
            _group_count(pooled_channels), pooled_channels
        )
        self.fuse = ConvGNAct(
            pooled_channels, channels, 1, padding=0, activation=False
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        reduced = self.reduce(x)

        horizontal = F.adaptive_avg_pool2d(reduced, (height, 1))
        horizontal = self.horizontal_norm(self.horizontal_conv(horizontal))
        horizontal = self.activation(horizontal).expand(-1, -1, -1, width)

        vertical = F.adaptive_avg_pool2d(reduced, (1, width))
        vertical = self.vertical_norm(self.vertical_conv(vertical))
        vertical = self.activation(vertical).expand(-1, -1, height, -1)

        context = self.fuse(self.activation(horizontal + vertical))
        return self.activation(x + context)


class RoadReconstructionDecoder(nn.Module):
    """S8-to-S1 road reconstruction decoder."""

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
            RepDepthwiseBlock(s4_channels, deploy=deploy),
            RepDepthwiseBlock(s4_channels, deploy=deploy),
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
            RepDepthwiseBlock(s2_channels, deploy=deploy),
            RepDepthwiseBlock(s2_channels, deploy=deploy),
        )

        self.full_refine = SeparableConvBNAct(s2_channels, full_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(full_channels, num_classes, 1)

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        stem_s2: Tensor,
        shallow_s4: Tensor,
        fused_s8: Tensor,
        output_size: Tuple[int, int],
    ) -> Tensor:
        p4 = self._resize(self.fused_proj(fused_s8), shallow_s4.shape[-2:])
        p4 = self.s4_fuse(torch.cat((p4, self.shallow_proj(shallow_s4)), dim=1))
        p4 = self.s4_refine(p4)

        p2 = self._resize(p4, stem_s2.shape[-2:])
        p2 = self.s2_fuse(torch.cat((p2, self.stem_proj(stem_s2)), dim=1))
        p2 = self.s2_refine(p2)

        full = self._resize(p2, output_size)
        return self.classifier(self.dropout(self.full_refine(full)))

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, (RepVGGBlock, RepDepthwiseBlock)):
                module.switch_to_deploy()


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


def soft_cldice_loss(
    probability: Tensor,
    target: Tensor,
    iterations: int = 8,
    smooth: float = 1e-6,
    downsample: int = 1,
) -> Tensor:
    """Soft clDice topology term (Shit et al., CVPR 2021, "clDice").

    Skeletonizing the *prediction* and checking how much of it lies inside
    the target mask gives a topological precision; skeletonizing the
    *target* and checking how much of it lies inside the predicted mask
    gives a topological sensitivity/recall. A single pixel gap that
    disconnects a road is penalized here even when it barely changes total
    area overlap -- exactly the failure mode plain Dice is blind to for
    thin, elongated structures.

    clDice by itself can be gamed by a prediction whose skeleton happens to
    thread through the target mask without covering its true width or
    shape, so this term is meant to be added on top of a region loss
    (Dice/CE), never used alone -- see RoadSegClDiceLoss.

    Unlike the old centerline-Tversky auxiliary, this term skeletonizes the
    *prediction* too, and does so with gradients (the target's skeleton is
    fixed and detached below). ``soft_skeletonize`` costs several max-pool
    passes over the full tensor per iteration, so doing this at full
    training resolution on every step is the single most expensive line in
    the loss. ``downsample`` (an integer stride, e.g. 4) shrinks the map
    with avg/max pooling before skeletonizing -- a road that survives to S4
    is still recognizably connected or broken, and the corresponding compute
    drops with the square of the factor. Set 1 to skip this and skeletonize
    at the input resolution exactly as given.
    """
    probability = probability.float()
    target = target.float()
    if downsample > 1:
        probability = F.avg_pool2d(probability, downsample)
        with torch.no_grad():
            target = F.max_pool2d(target, downsample)
    skeleton_pred = soft_skeletonize(probability, iterations)
    with torch.no_grad():
        skeleton_true = soft_skeletonize(target, iterations)

    skeleton_pred_flat = skeleton_pred.flatten(1)
    skeleton_true_flat = skeleton_true.flatten(1)
    probability_flat = probability.flatten(1)
    target_flat = target.flatten(1)

    precision = (skeleton_pred_flat * target_flat).sum(dim=1) + smooth
    precision = precision / (skeleton_pred_flat.sum(dim=1) + smooth)
    sensitivity = (skeleton_true_flat * probability_flat).sum(dim=1) + smooth
    sensitivity = sensitivity / (skeleton_true_flat.sum(dim=1) + smooth)

    cl_dice = 2.0 * precision * sensitivity / (precision + sensitivity + smooth)
    return (1.0 - cl_dice).mean()


class RoadSegClDiceLoss(nn.Module):
    """Road objective: weighted CE + clDice, plus S16 deep supervision.

    The main term is exactly two losses: weighted CE for per-pixel class
    balance, and soft-clDice (Shit et al., CVPR 2021) for road topology --
    no separate area-based Dice term. This replaces the previous S4
    centerline-Tversky auxiliary head: clDice already supervises topology
    directly on the full-resolution prediction, so a separate
    skeleton-prediction head is redundant and has been removed from the
    decoder.

    Caution carried over from the clDice paper: without a region/area term
    like Dice, a prediction whose skeleton threads through the target mask
    can score well on clDice while still being too thin, too thick, or
    locally mis-shaped -- clDice constrains connectivity, not area. Watch
    fixed@.50 IoU (an area metric) alongside F1/relaxed-F1 during training;
    if IoU stalls or drops while topology-sensitive metrics keep improving,
    that is the signature of this failure mode, and raising cldice_weight
    won't fix it -- a Dice/Tversky term would need to come back.

    A different auxiliary signal is unchanged from before: a lightweight
    road head reading the *semantic* S16 feature inside DualResolutionContext
    (after the DAPPM S32 context has been folded back in), supervised
    against a max-pooled downsample of the target. This is deep supervision
    for the semantic branch specifically -- the cross-branch gate statistics
    logged during training showed the semantic->detail exchange staying an
    order of magnitude stronger than the reverse detail->semantic direction,
    i.e. the semantic stream had little direct incentive to become
    road-discriminative on its own. A cheap, ungated CE+Dice loss on that
    stream gives it a direct reason to; that internal Dice term is
    unaffected by the change above since it supervises a different, much
    coarser (S16) prediction, not the main output.
    """

    def __init__(
        self,
        road_class_weight: float = 2.0,
        cldice_weight: float = 0.5,
        skeleton_iterations: int = 8,
        cldice_downsample: int = 4,
        semantic_aux_weight: float = 0.15,
        semantic_aux_dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.road_class_weight = float(road_class_weight)
        self.cldice_weight = float(cldice_weight)
        self.skeleton_iterations = int(skeleton_iterations)
        self.cldice_downsample = int(cldice_downsample)
        self.semantic_aux_weight = float(semantic_aux_weight)
        self.semantic_aux_dice_weight = float(semantic_aux_dice_weight)

    def forward(
        self,
        outputs: Tuple[Tensor, Tensor],
        target: Tensor,
    ) -> Dict[str, Tensor]:
        semantic_aux_logits, road_logits = outputs
        labels = (target > 0).long()
        road_mask = labels.unsqueeze(1).float()
        class_weights = road_logits.new_tensor([1.0, self.road_class_weight])

        loss_main_ce = F.cross_entropy(
            road_logits.float(), labels, weight=class_weights
        )
        road_probability = road_logits.float().softmax(dim=1)[:, 1:2]
        loss_main_cldice = soft_cldice_loss(
            road_probability,
            road_mask,
            self.skeleton_iterations,
            downsample=self.cldice_downsample,
        )

        with torch.no_grad():
            # Max-pool, not average-pool: a thin road must survive being
            # downsampled to S16 the same way it had to survive to S4 in the
            # old centerline target -- averaging fades it below the
            # rounding threshold in cells that are mostly background.
            aux_target = F.adaptive_max_pool2d(
                road_mask, semantic_aux_logits.shape[-2:]
            )
        aux_labels = aux_target.squeeze(1).long()
        aux_class_weights = semantic_aux_logits.new_tensor(
            [1.0, self.road_class_weight]
        )
        loss_aux_ce = F.cross_entropy(
            semantic_aux_logits.float(), aux_labels, weight=aux_class_weights
        )
        aux_probability = semantic_aux_logits.float().softmax(dim=1)[:, 1:2]
        loss_aux_dice = binary_dice_loss(aux_probability, aux_target)
        loss_semantic_aux = (
            loss_aux_ce + self.semantic_aux_dice_weight * loss_aux_dice
        )

        total = (
            loss_main_ce
            + self.cldice_weight * loss_main_cldice
            + self.semantic_aux_weight * loss_semantic_aux
        )
        return {
            "loss_total": total,
            "loss_main_ce": loss_main_ce.detach(),
            "loss_main_cldice": loss_main_cldice.detach(),
            "loss_semantic_aux": loss_semantic_aux.detach(),
        }


@torch.no_grad()
def verify_reparameterization(
    block: Union[RepVGGBlock, RepDepthwiseBlock],
    shape: Tuple[int, int, int, int],
) -> float:
    """Return max absolute output error before and after branch fusion."""
    block.eval()
    x = torch.randn(shape, device=next(block.parameters()).device)
    reference = block(x)
    block.switch_to_deploy()
    return float((reference - block(x)).abs().max())
