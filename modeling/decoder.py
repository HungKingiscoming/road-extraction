from __future__ import annotations

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



class GlobalRoadContext(nn.Module):
    """Lightweight full-CNN global context block for S8 road features.

    Four complementary branches capture local context, two larger receptive
    fields, and image-level scene context. The fused residual is zero-initialized
    so the block starts as an exact identity mapping and can be added safely to
    an already-trained model.
    """

    def __init__(
        self,
        channels: int,
        branch_channels: Optional[int] = None,
        dilations: Tuple[int, int] = (2, 4),
    ) -> None:
        super().__init__()
        branch_channels = branch_channels or max(16, channels // 4)

        def spatial_branch(dilation: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                    groups=channels,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )

        self.local = spatial_branch(1)
        self.dilated_1 = spatial_branch(int(dilations[0]))
        self.dilated_2 = spatial_branch(int(dilations[1]))
        # Avoid BatchNorm after 1x1 global pooling so batch-size 1 remains valid.
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, branch_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_channels * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.fuse[-1].weight)
        nn.init.zeros_(self.fuse[-1].bias)

    def forward(self, x: Tensor) -> Tensor:
        target_size = x.shape[-2:]
        global_context = F.interpolate(
            self.global_pool(x), target_size, mode="bilinear", align_corners=False
        )
        context = torch.cat(
            (
                self.local(x),
                self.dilated_1(x),
                self.dilated_2(x),
                global_context,
            ),
            dim=1,
        )
        return x + self.fuse(context)

def _group_count(channels: int, maximum: int = 8) -> int:
    """Largest small GroupNorm divisor, robust for small per-GPU batches."""
    for groups in range(min(maximum, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


class SkipFeatureGate(nn.Module):
    """Semantically-guided skip-connection denoising (full-CNN SSFRM-lite).

    Shallow ResNet stem/layer1 features carry rich edges but also background
    texture that looks locally road-like (driveways, rooftops, parking lots).
    Concatenating them into the decoder unfiltered lets that noise leak into
    the prediction.  This gate lets the already-decoded feature (semantically
    deeper, since it has passed through the fused detail/semantic branch)
    suppress that noise before the concatenation: one squeeze-excite style
    channel gate plus one spatial gate, both driven by the decoder feature.
    This targets the same problem as SSFRM (Yang et al., TGRS 2026) --
    channel- and spatial-dimension refinement of skip features under deep
    semantic guidance -- using only conv/pool/sigmoid instead of a learned
    channel/spatial similarity matrix, so it stays attention-free.  Both
    gates are zero-initialized to output exactly 1 (via ``2 * sigmoid(0)``),
    so training starts identical to an unfiltered skip connection.
    """

    def __init__(self, skip_channels: int, guide_channels: int) -> None:
        super().__init__()
        hidden = max(8, skip_channels // 4)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(guide_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, skip_channels, 1),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(
                guide_channels + skip_channels, hidden, 3, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
        nn.init.zeros_(self.spatial_gate[-1].weight)
        nn.init.zeros_(self.spatial_gate[-1].bias)

    def forward(self, skip: Tensor, guide: Tensor) -> Tensor:
        if skip.shape[-2:] != guide.shape[-2:]:
            raise ValueError("SkipFeatureGate inputs must be spatially aligned")
        channel_weight = 2.0 * torch.sigmoid(self.channel_gate(guide))
        spatial_weight = 2.0 * torch.sigmoid(
            self.spatial_gate(torch.cat((skip, guide), dim=1))
        )
        return skip * channel_weight * spatial_weight


_SKIP_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
).view(1, 1, 3, 3)



class DirectionHead(nn.Module):
    """Predict a unit 2-D road direction from skip and semantic guide features.

    No direction target or auxiliary loss is required. Because aggregation samples
    both +d and -d, the sign of the direction is irrelevant, which naturally
    matches the 180-degree symmetry of roads.
    """

    def __init__(self, skip_channels: int, guide_channels: int) -> None:
        super().__init__()
        hidden = max(16, skip_channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(
                skip_channels + guide_channels,
                hidden,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1),
        )
        # Small non-zero initialization avoids the undefined zero-vector case
        # while the residual gate keeps the whole module close to identity.
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, skip: Tensor, guide: Tensor) -> Tensor:
        direction = self.net(torch.cat((skip, guide), dim=1))
        norm = torch.sqrt((direction * direction).sum(dim=1, keepdim=True) + 1e-6)
        return direction / norm


class OrientedSkipAggregation(nn.Module):
    """Self-supervised road-direction-steered skip refinement.

    Direction is learned only through the segmentation objective (BCE + clDice).
    Gradients are allowed to flow through ``grid_sample`` into ``DirectionHead``.
    Context is sampled symmetrically along the predicted road axis, so no explicit
    orientation label is needed.
    """

    def __init__(
        self,
        skip_channels: int,
        guide_channels: int,
        span: int = 2,
        spacing: float = 3.0,
    ) -> None:
        super().__init__()
        self.span = int(span)
        self.spacing = float(spacing)
        self.direction = DirectionHead(skip_channels, guide_channels)
        hidden = max(16, skip_channels // 4)
        self.confidence = nn.Sequential(
            nn.Conv2d(guide_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(skip_channels * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, skip_channels, 1),
        )
        nn.init.zeros_(self.confidence[-1].weight)
        # A small initial residual lets segmentation gradients reach the
        # direction head immediately without strongly perturbing pretrained skips.
        nn.init.constant_(self.confidence[-1].bias, -2.0)

    def forward(self, skip: Tensor, guide: Tensor) -> Tensor:
        if skip.shape[-2:] != guide.shape[-2:]:
            raise ValueError("OrientedSkipAggregation inputs must be spatially aligned")

        batch, _, height, width = skip.shape
        direction = self.direction(skip, guide)

        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=skip.device, dtype=skip.dtype),
            torch.linspace(-1.0, 1.0, width, device=skip.device, dtype=skip.dtype),
            indexing="ij",
        )
        base = torch.stack((xs, ys), dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

        aggregated = torch.zeros_like(skip)
        x_scale = 2.0 / max(width - 1, 1)
        y_scale = 2.0 / max(height - 1, 1)
        for step in range(1, self.span + 1):
            distance = self.spacing * float(step)
            offset = torch.cat(
                (
                    direction[:, 0:1] * (distance * x_scale),
                    direction[:, 1:2] * (distance * y_scale),
                ),
                dim=1,
            )
            for sign in (1.0, -1.0):
                grid = (base + sign * offset).permute(0, 2, 3, 1)
                aggregated = aggregated + F.grid_sample(
                    skip,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
        aggregated = aggregated / float(max(self.span * 2, 1))

        confidence = torch.sigmoid(self.confidence(guide))
        context = self.fuse(torch.cat((skip, aggregated), dim=1))
        return skip + confidence * context

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
    """Light S8-to-S1 road decoder with segmentation-trained directional skips.

    Global context is disabled by default because DualResolutionContext already
    injects S32 dilated semantic context before this decoder.
    """

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
        full_refine_blocks: int = 0,
        oriented_skip: bool = True,
        oriented_skip_span: int = 2,
        oriented_skip_spacing: float = 3.0,
        use_global_context: bool = False,
        global_context_dilations: Tuple[int, int] = (2, 4),
        deploy: bool = False,
    ) -> None:
        super().__init__()
        shallow_skip_channels = max(24, s4_channels // 2)
        stem_skip_channels = max(16, s2_channels // 2)

        self.global_context = (
            GlobalRoadContext(
                fused_channels,
                dilations=global_context_dilations,
            )
            if use_global_context
            else nn.Identity()
        )

        skip_refine_cls = OrientedSkipAggregation if oriented_skip else SkipFeatureGate
        skip_refine_kwargs = (
            {"span": oriented_skip_span, "spacing": oriented_skip_spacing}
            if oriented_skip
            else {}
        )

        self.fused_proj = ConvBNAct(fused_channels, s4_channels, 1, padding=0)
        self.shallow_proj = ConvBNAct(
            shallow_channels, shallow_skip_channels, 1, padding=0
        )
        self.s4_skip_gate = skip_refine_cls(
            shallow_skip_channels, s4_channels, **skip_refine_kwargs
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

        self.stem_proj = ConvBNAct(stem_channels, stem_skip_channels, 1, padding=0)
        self.s2_skip_gate = skip_refine_cls(
            stem_skip_channels, s4_channels, **skip_refine_kwargs
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
        self.full_extra_refine = (
            nn.Sequential(
                *[
                    RepDepthwiseBlock(full_channels, deploy=deploy)
                    for _ in range(int(full_refine_blocks))
                ]
            )
            if full_refine_blocks > 0
            else nn.Identity()
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Conv2d(full_channels, num_classes, 1)

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    @staticmethod
    def _decode_stage(
        upsampled: Tensor,
        raw_skip: Tensor,
        skip_proj: nn.Module,
        skip_refine: nn.Module,
        fuse: nn.Module,
        refine: nn.Module,
    ) -> Tensor:
        skip = skip_refine(skip_proj(raw_skip), upsampled)
        fused = fuse(torch.cat((upsampled, skip), dim=1))
        return refine(fused)

    def forward(
        self,
        stem_s2: Tensor,
        shallow_s4: Tensor,
        fused_s8: Tensor,
        output_size: Tuple[int, int],
    ) -> Tensor:
        # Global scene/road context is injected only at S8, then the existing
        # coarse-to-fine decoder handles precise spatial reconstruction.
        fused_s8 = self.global_context(fused_s8)

        p4 = self._resize(self.fused_proj(fused_s8), shallow_s4.shape[-2:])
        p4 = self._decode_stage(
            p4,
            shallow_s4,
            self.shallow_proj,
            self.s4_skip_gate,
            self.s4_fuse,
            self.s4_refine,
        )

        p2 = self._resize(p4, stem_s2.shape[-2:])
        p2 = self._decode_stage(
            p2,
            stem_s2,
            self.stem_proj,
            self.s2_skip_gate,
            self.s2_fuse,
            self.s2_refine,
        )

        full = self._resize(p2, output_size)
        full = self.full_extra_refine(self.full_refine(full))
        road_logits = self.classifier(self.dropout(full))
        # The decoder now always returns segmentation logits. DirectionHead is
        # optimized implicitly through BCE + clDice via differentiable grid_sample.
        return road_logits

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, (RepVGGBlock, RepDepthwiseBlock)):
                module.switch_to_deploy()

def _soft_erode(x: Tensor) -> Tensor:
    """Cross-shaped (not square) morphological erosion.

    A full 3x3 erosion would shrink a road already only a few pixels wide
    almost as fast as it shrinks the background it's supposed to remove --
    two 1-pixel-wide (3x1 and 1x3) erosions, combined with min, approximate
    a disk-shaped structuring element that thins more gently.
    """
    p1 = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    p2 = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(x: Tensor) -> Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def _soft_open(x: Tensor) -> Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeletonize(x: Tensor, iterations: int) -> Tensor:
    """Differentiable soft skeleton (Shit et al., clDice, CVPR 2021).

    A morphological opening restores everything erosion removed except the
    thin centerline; repeatedly eroding and taking what opening can no
    longer restore accumulates that centerline into a skeleton map, entirely
    with maxpool/minpool so it stays differentiable end to end.
    """
    skeleton = F.relu(x - _soft_open(x))
    eroded = x
    for _ in range(iterations):
        eroded = _soft_erode(eroded)
        opened = _soft_open(eroded)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice_loss(
    probability: Tensor,
    target: Tensor,
    iterations: int = 10,
    eps: float = 1e-6,
) -> Tensor:
    """1 - clDice: topology-preserving loss for tubular/curvilinear structures
    (Shit et al., "clDice -- A Novel Topology-Preserving Loss Function for
    Tubular Structure Segmentation", CVPR 2021).

    Plain Dice scores overlap over the *whole* road area, so a broken
    connection (a short missing segment) barely moves the score if the
    surrounding road pixels are still correct -- exactly the failure mode
    that tanks a road network's usefulness without tanking Dice/IoU much.
    clDice instead measures overlap against each side's *soft skeleton*
    (topology precision: how much of the predicted skeleton lies on real
    road; topology sensitivity: how much of the true skeleton is covered by
    the prediction), so a disconnection is penalized in proportion to how
    much of the route it breaks, not how many pixels it spans.
    """
    probability = probability.float()
    target = target.float()
    skeleton_pred = soft_skeletonize(probability, iterations)
    with torch.no_grad():
        skeleton_true = soft_skeletonize(target, iterations)
    precision = (skeleton_pred * target).sum(dim=(1, 2, 3))
    precision = (precision + eps) / (skeleton_pred.sum(dim=(1, 2, 3)) + eps)
    sensitivity = (skeleton_true * probability).sum(dim=(1, 2, 3))
    sensitivity = (sensitivity + eps) / (skeleton_true.sum(dim=(1, 2, 3)) + eps)
    cl_dice = 2.0 * precision * sensitivity / (precision + sensitivity + eps)
    return (1.0 - cl_dice).mean()



class RoadSegBCEClDiceLoss(nn.Module):
    """Exactly two training losses: weighted BCE + clDice.

    ``cldice_weight`` scales the topology term directly; it is no longer a
    Dice/clDice blend coefficient because plain Dice is intentionally removed.
    """

    def __init__(
        self,
        road_class_weight: float = 2.0,
        bce_weight: float = 1.0,
        cldice_weight: float = 1.0,
        cldice_iterations: int = 10,
    ) -> None:
        super().__init__()
        self.road_class_weight = float(road_class_weight)
        self.bce_weight = float(bce_weight)
        self.cldice_weight = float(cldice_weight)
        self.cldice_iterations = int(cldice_iterations)

    @staticmethod
    def _extract_logits(outputs: Union[Tensor, Tuple]) -> Tensor:
        # Supports the new decoder (Tensor) and old training checkpoints/code
        # that may still wrap logits in a tuple.
        if isinstance(outputs, Tensor):
            return outputs
        if isinstance(outputs, tuple):
            for item in reversed(outputs):
                if isinstance(item, Tensor):
                    return item
        raise TypeError("Expected road logits Tensor or a tuple containing one")

    def forward(
        self,
        outputs: Union[Tensor, Tuple],
        target: Tensor,
    ) -> Dict[str, Tensor]:
        road_logits = self._extract_logits(outputs)
        labels = (target > 0).long()
        road_mask = labels.unsqueeze(1).float()

        # For two logits [background, road], BCE on (road - background) is
        # exactly equivalent to two-class softmax cross entropy.
        road_logit = road_logits.float()[:, 1] - road_logits.float()[:, 0]
        loss_bce = F.binary_cross_entropy_with_logits(
            road_logit,
            road_mask[:, 0],
            pos_weight=road_logits.new_tensor(self.road_class_weight),
        )

        road_probability = torch.sigmoid(road_logit).unsqueeze(1)
        loss_cldice = soft_cldice_loss(
            road_probability,
            road_mask,
            iterations=self.cldice_iterations,
        )

        total = self.bce_weight * loss_bce + self.cldice_weight * loss_cldice
        zero = loss_bce.detach().new_zeros(())
        return {
            "loss_total": total,
            "loss_bce": loss_bce.detach(),
            "loss_cldice": loss_cldice.detach(),
            # Legacy logging aliases only; neither contributes to loss_total.
            "loss_main_bce": loss_bce.detach(),
            "loss_main_dice": zero,
            "loss_aux_orientation": zero,
        }


class RoadSegOrientationLoss(RoadSegBCEClDiceLoss):
    """Backward-compatible name; orientation/Dice losses are not used anymore.

    ``main_dice_weight`` and ``aux_weight`` are accepted only so an older
    training script does not fail at construction time. They have no effect.
    """

    def __init__(
        self,
        road_class_weight: float = 2.0,
        main_dice_weight: float = 1.0,
        aux_weight: float = 0.0,
        cldice_weight: float = 1.0,
        cldice_iterations: int = 10,
        bce_weight: float = 1.0,
    ) -> None:
        super().__init__(
            road_class_weight=road_class_weight,
            bce_weight=bce_weight,
            cldice_weight=cldice_weight,
            cldice_iterations=cldice_iterations,
        )
        # Kept as inert attributes for old warmup/logging code that may access them.
        self.main_dice_weight = 0.0
        self.aux_weight = 0.0


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
