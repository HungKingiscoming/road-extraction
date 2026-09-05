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


def structure_tensor_angle(
    mask: Tensor, sobel_x: Tensor, eps: float = 1e-6
) -> Tuple[Tensor, Tensor, Tensor]:
    """Local road-tangent direction (double-angle form) from a binary mask.

    Ground-truth counterpart to ``OrientHead``'s prediction, used to
    supervise it. Sobel gradients of the mask feed a structure-tensor
    (second-moment matrix); its dominant axis, in double-angle form
    ``(p, q) = (cos(2*theta), sin(2*theta))``, is agnostic to a road's
    180-degree undirected symmetry (a road pointing left-to-right and one
    pointing right-to-left are the same line). That axis is the
    gradient/normal direction (perpendicular to the road); rotating 90
    degrees -- negating (p, q), since doubling a 90-degree rotation gives
    180 degrees -- gives the tangent (along-road) direction actually wanted.
    Returns unit-normalized ``(p, q)`` plus a ``valid`` mask (low-gradient
    regions and background pixels have no reliable/meaningful direction).
    """
    mask = mask[:, 0].float() if mask.ndim == 4 else mask.float()
    ix = F.conv2d(mask.unsqueeze(1), sobel_x, padding=1)[:, 0]
    iy = F.conv2d(mask.unsqueeze(1), sobel_x.transpose(2, 3), padding=1)[:, 0]
    jxx = F.avg_pool2d((ix * ix).unsqueeze(1), 5, 1, 2)[:, 0]
    jyy = F.avg_pool2d((iy * iy).unsqueeze(1), 5, 1, 2)[:, 0]
    jxy = F.avg_pool2d((ix * iy).unsqueeze(1), 5, 1, 2)[:, 0]
    # Normal (gradient) axis, rotated 90 degrees (negated) to the tangent.
    p, q = -(jxx - jyy), -(2.0 * jxy)
    n = torch.sqrt(p * p + q * q)
    valid = (n > 5e-3).float() * (mask > 0.5).float()
    n = n + eps
    return (p / n).unsqueeze(1), (q / n).unsqueeze(1), valid.unsqueeze(1)


class OrientHead(nn.Module):
    """Predicts (p, q, confidence) = (cos2theta, sin2theta, confidence) per pixel."""

    def __init__(self, channels: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or max(16, channels // 2)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )
        # Zero-init: starts with a single, stable (if arbitrary) direction
        # over the whole feature map and confidence=sigmoid(0)=0.5, instead
        # of a noisy random direction per pixel on step 1.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        p, q, confidence = self.net(x).chunk(3, dim=1)
        n = torch.sqrt(p * p + q * q + 1e-6)
        return p / n, q / n, torch.sigmoid(confidence)


class OrientedSkipAggregation(nn.Module):
    """Road-direction-steered skip-feature refinement (learned, supervised direction).

    Drop-in alternative to SkipFeatureGate -- same
    ``(projected_skip, guide) -> same-shape tensor`` signature expected by
    ``RoadReconstructionDecoder._decode_stage`` -- implementing RoadWeaveNet's
    WeaveAgg/OCFE mechanism (Orientation-Conditioned Feature Extraction):
    instead of gating the skip feature channel/spatial-wise, it aggregates
    context sampled *along the skip feature's own local road direction*,
    which is the right receptive-field shape for a road (long and thin) in a
    way an isotropic conv or a plain gate cannot express.

    Direction is predicted by a small trained head (``OrientHead``) rather
    than computed analytically, so it can adapt beyond what a fixed
    Sobel/structure-tensor estimate of the raw activations would give. That
    head's own gradient signal comes only from an external auxiliary loss
    against ``structure_tensor_angle`` of the ground-truth mask (see
    ``last_orientation`` below) -- its predicted direction is detached
    before building the sampling grid here, since grid_sample's gradient
    through a geometric sampling location is too weak/indirect to train a
    direction head on its own. The confidence gate and fuse conv remain
    purely segmentation-loss-trained, same as SkipFeatureGate.
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
        self.orient = OrientHead(skip_channels)
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
        # sigmoid(-4) ~= 0.018: starts as a near-identity skip connection
        # (small but nonzero, so gradient can still reach and grow it)
        # rather than the exact 2*sigmoid(0)=1 identity trick used elsewhere
        # in this file -- here confidence scales an added residual, not a
        # multiplicative pass-through, so identity means ~0, not 1.
        nn.init.constant_(self.confidence[-1].bias, -4.0)
        # Last (p, q) prediction, kept (not detached) for an external loss to
        # supervise against structure_tensor_angle(mask) -- see
        # RoadReconstructionDecoder.forward and RoadSegOrientationLoss. Not a
        # registered buffer: this is per-step scratch, not model state.
        self.last_orientation: Optional[Tuple[Tensor, Tensor]] = None

    def forward(self, skip: Tensor, guide: Tensor) -> Tensor:
        if skip.shape[-2:] != guide.shape[-2:]:
            raise ValueError(
                "OrientedSkipAggregation inputs must be spatially aligned"
            )
        height, width = skip.shape[-2:]
        p, q, _ = self.orient(skip)
        self.last_orientation = (p, q)
        direction = torch.cat((p, q), dim=1).detach()
        theta = 0.5 * torch.atan2(direction[:, 1:2], direction[:, 0:1])
        direction = torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=skip.device),
            torch.linspace(-1.0, 1.0, width, device=skip.device),
            indexing="ij",
        )
        base = torch.stack((xs, ys), dim=0).unsqueeze(0)  # 1,2,H,W

        aggregated = torch.zeros_like(skip)
        for step in range(1, self.span + 1):
            offset = self.spacing * step / width
            for sign in (1.0, -1.0):
                grid = (base + sign * offset * direction).permute(0, 2, 3, 1)
                aggregated = aggregated + F.grid_sample(
                    skip, grid, mode="bilinear", padding_mode="border",
                    align_corners=True,
                )
        aggregated = aggregated / float(self.span * 2)

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
    """S8-to-S1 decoder. In training mode with oriented_skip on, also
    surfaces both OrientedSkipAggregation stages' predicted road direction
    (S4, S2) for RoadSegOrientationLoss to supervise -- both must be
    supervised, since each has an OrientHead that would otherwise never
    receive gradient (its direction output is detached before use)."""

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
        deploy: bool = False,
    ) -> None:
        super().__init__()
        shallow_skip_channels = max(24, s4_channels // 2)
        stem_skip_channels = max(16, s2_channels // 2)
        # Both S4 and S2 skip stages use the same kind of refinement module,
        # picked once here. OrientedSkipAggregation (road-direction-steered,
        # direction learned and supervised via RoadSegOrientationLoss) is the
        # default; SkipFeatureGate (channel/spatial SE-style denoising) stays
        # available via --no-oriented_skip for a clean A/B comparison
        # without another code change. OrientedSkipAggregation runs S4/S2
        # grid_sample + several conv layers per stage (not just SE-style
        # pooling), so it costs more VRAM than SkipFeatureGate at the same
        # batch size -- span/spacing (fewer/closer sampling offsets) are
        # exposed here too, so that cost is tunable without another one.
        skip_refine_cls = (
            OrientedSkipAggregation if oriented_skip else SkipFeatureGate
        )
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

        self.stem_proj = ConvBNAct(
            stem_channels, stem_skip_channels, 1, padding=0
        )
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
        # Optional extra depth at the full-resolution stage, targeting the
        # persistent gap between relaxed (+/-3px) and strict F1 (the model
        # locates roads correctly; it under-refines their exact edges). This
        # stage runs at the *largest* spatial size in the decoder, so unlike
        # every other RepDepthwiseBlock stack here its compute/VRAM cost is
        # real (roughly 4x/16x an S4/S2 block) regardless of full_channels --
        # default 0 keeps the original single-block full_refine so a plain
        # run fits the same batch size as before; opt in via
        # --full_refine_blocks once there is VRAM budget to test it.
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
        """One coarse-to-fine stage: project + refine the skip feature under
        guidance from the already-decoded ``upsampled`` feature, fuse the
        two, then refine the result.

        S4 and S2 both follow this exact pattern (only the submodules
        differ), so it is factored out once here rather than duplicated.
        ``skip_refine`` is deliberately generic -- ``OrientedSkipAggregation``
        (road-direction-steered context aggregation) by default, or
        ``SkipFeatureGate`` (channel+spatial SE-style denoising) via
        ``oriented_skip=False`` -- both implement
        ``(projected_skip, upsampled) -> same-shape tensor``, so switching
        between them (or trying a third option later) means picking a class
        in ``__init__``, not touching this method or ``forward``.
        """
        skip = skip_refine(skip_proj(raw_skip), upsampled)
        fused = fuse(torch.cat((upsampled, skip), dim=1))
        return refine(fused)

    def forward(
        self,
        stem_s2: Tensor,
        shallow_s4: Tensor,
        fused_s8: Tensor,
        output_size: Tuple[int, int],
    ) -> Union[
        Tensor,
        Tuple[Tuple[Optional[Tuple[Tensor, Tensor]], Optional[Tuple[Tensor, Tensor]]], Tensor],
    ]:
        p4 = self._resize(self.fused_proj(fused_s8), shallow_s4.shape[-2:])
        p4 = self._decode_stage(
            p4, shallow_s4, self.shallow_proj, self.s4_skip_gate,
            self.s4_fuse, self.s4_refine,
        )

        p2 = self._resize(p4, stem_s2.shape[-2:])
        p2 = self._decode_stage(
            p2, stem_s2, self.stem_proj, self.s2_skip_gate,
            self.s2_fuse, self.s2_refine,
        )

        full = self._resize(p2, output_size)
        full = self.full_extra_refine(self.full_refine(full))
        road_logits = self.classifier(self.dropout(full))
        if self.training:
            # last_orientation is None whenever oriented_skip=False (plain
            # SkipFeatureGate has no direction to supervise). Both stages
            # must be supervised, not just one: OrientHead's predicted
            # direction is detached before use in grid_sample (see
            # OrientedSkipAggregation), so a stage whose orientation is
            # never returned to the loss would have an OrientHead that
            # never receives *any* gradient -- silently dead weight, and
            # fatal under DDP (which errors on parameters with no grad).
            orientations = (
                getattr(self.s4_skip_gate, "last_orientation", None),
                getattr(self.s2_skip_gate, "last_orientation", None),
            )
            return orientations, road_logits
        return road_logits

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, (RepVGGBlock, RepDepthwiseBlock)):
                module.switch_to_deploy()


def binary_dice_loss(
    probability: Tensor, target: Tensor, eps: float = 1e-6
) -> Tensor:
    probability = probability.float().flatten(1)
    target = target.float().flatten(1)
    intersection = (probability * target).sum(dim=1)
    denominator = probability.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


class RoadSegOrientationLoss(nn.Module):
    """Road objective: weighted CE + Dice + road-orientation supervision.

    The orientation term supervises both OrientedSkipAggregation stages'
    OrientHead (S4 and S2) against ``structure_tensor_angle`` of the
    ground-truth mask, each resized to that stage's predicted (p, q)
    resolution, then averaged. Both stages must be supervised: an
    OrientHead's predicted direction is detached before use in
    grid_sample (see OrientedSkipAggregation), so a stage left out here
    would have a head that never receives any gradient at all -- silently
    dead weight, and fatal under DDP (which errors on parameters with no
    grad). This term occupies the same slot, warmup schedule
    (``aux_start_epoch``/``aux_warmup_epochs``), and weight (``aux_weight``)
    previously used for centerline Tversky supervision, which it replaces
    rather than adds to a fourth loss term. When ``oriented_skip=False``
    (plain SkipFeatureGate, no direction to supervise), ``outputs[0]``'s
    entries are ``None`` and this term is simply zero.
    """

    def __init__(
        self,
        road_class_weight: float = 2.0,
        main_dice_weight: float = 1.0,
        aux_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.road_class_weight = float(road_class_weight)
        self.main_dice_weight = float(main_dice_weight)
        self.aux_weight = float(aux_weight)
        self.register_buffer("_sobel_x", _SKIP_SOBEL_X, persistent=False)

    def _orientation_term(
        self, orientation: Optional[Tuple[Tensor, Tensor]], road_mask: Tensor
    ) -> Optional[Tensor]:
        if orientation is None:
            return None
        p_pred, q_pred = orientation
        with torch.no_grad():
            target_mask = F.adaptive_max_pool2d(road_mask, p_pred.shape[-2:])
            p_gt, q_gt, valid = structure_tensor_angle(target_mask, self._sobel_x)
        # cos(2 * (theta_pred - theta_gt)); 1 minus that is 0 when the
        # predicted and ground-truth directions are aligned (mod 180deg).
        alignment = p_pred * p_gt + q_pred * q_gt
        valid_count = valid.sum().clamp_min(1.0)
        return ((1.0 - alignment) * valid).sum() / valid_count

    def forward(
        self,
        outputs: Tuple[Tuple[Optional[Tuple[Tensor, Tensor]], ...], Tensor],
        target: Tensor,
    ) -> Dict[str, Tensor]:
        orientations, road_logits = outputs
        labels = (target > 0).long()
        road_mask = labels.unsqueeze(1).float()
        class_weights = road_logits.new_tensor([1.0, self.road_class_weight])
        loss_main_ce = F.cross_entropy(
            road_logits.float(), labels, weight=class_weights
        )
        road_probability = road_logits.float().softmax(dim=1)[:, 1:2]
        loss_main_dice = binary_dice_loss(road_probability, road_mask)

        # Always computed (even when aux_weight==0.0 during warmup), same as
        # the pre-warmup centerline loss this replaced: a term that is
        # skipped outright rather than multiplied by a zero weight never
        # touches OrientHead's parameters in the backward graph at all, so
        # their gradient hooks never fire -- fine on a single GPU, but DDP
        # requires every parameter to participate (even with a zero
        # resulting gradient) every step, and errors out otherwise.
        terms = [
            self._orientation_term(orientation, road_mask)
            for orientation in orientations
        ]
        terms = [term for term in terms if term is not None]
        loss_orientation = (
            torch.stack(terms).mean() if terms else road_logits.new_zeros(())
        )

        total = (
            loss_main_ce
            + self.main_dice_weight * loss_main_dice
            + self.aux_weight * loss_orientation
        )
        return {
            "loss_total": total,
            "loss_main_ce": loss_main_ce.detach(),
            "loss_main_dice": loss_main_dice.detach(),
            "loss_aux_orientation": loss_orientation.detach(),
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
