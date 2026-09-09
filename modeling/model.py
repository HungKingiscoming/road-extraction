from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .decoder import (
    ConvBNAct,
    ConvGNAct,
    RepDepthwiseBlock,
    RepVGGBlock,
    RoadReconstructionDecoder,
    StripPoolingModule,
    _group_count,
)


def _extract_state_dict(checkpoint: object) -> Dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Encoder checkpoint must contain a state dictionary")
    for key in ("state_dict", "model", "ema"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    if not isinstance(checkpoint, dict):
        raise TypeError("Could not find a state dictionary")
    state: Dict[str, Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, Tensor):
            continue
        clean = str(key)
        for prefix in ("module.", "encoder.backbone.", "backbone."):
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
        state[clean] = value
    return state


def _build_resnet34(
    imagenet_pretrained: bool,
    encoder_weights_path: Optional[str],
) -> nn.Module:
    try:
        from torchvision.models import ResNet34_Weights, resnet34

        weights = (
            ResNet34_Weights.DEFAULT
            if imagenet_pretrained and not encoder_weights_path
            else None
        )
        backbone = resnet34(weights=weights)
    except ImportError as error:
        raise ImportError("torchvision is required for ResNet-34") from error
    except TypeError:
        from torchvision.models import resnet34

        backbone = resnet34(
            pretrained=bool(imagenet_pretrained and not encoder_weights_path)
        )

    if encoder_weights_path:
        path = Path(encoder_weights_path)
        if not path.is_file():
            raise FileNotFoundError(f"Encoder weights not found: {path}")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        state = _extract_state_dict(checkpoint)
        missing, _ = backbone.load_state_dict(state, strict=False)
        matched = len(backbone.state_dict()) - len(missing)
        if matched < 100:
            raise RuntimeError(
                f"Only {matched} ResNet tensors matched {path}; wrong weights?"
            )
    return backbone


class TruncatedResNet34(nn.Module):
    """Return pretrained ResNet features through layer4 (S32)."""

    out_channels = (64, 64, 128, 256, 512)

    def __init__(
        self,
        imagenet_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        backbone = _build_resnet34(
            imagenet_pretrained=imagenet_pretrained,
            encoder_weights_path=encoder_weights_path,
        )
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(
        self, x: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        stem_s2 = self.stem(x)
        shallow_s4 = self.layer1(self.maxpool(stem_s2))
        shared_s8 = self.layer2(shallow_s4)
        semantic_s16 = self.layer3(shared_s8)
        semantic_s32 = self.layer4(semantic_s16)
        return stem_s2, shallow_s4, shared_s8, semantic_s16, semantic_s32


class ProgressiveDAPPM(nn.Module):
    """Progressively aggregate adaptive pooled context at the semantic S32 map.

    GroupNorm keeps the global 1x1 branch valid for small per-GPU batches.
    Pooling proceeds from finer to coarser grids, so each stage adds broader
    context to the previous representation before concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        branch_channels: int,
        out_channels: int,
        pool_sizes: Sequence[int] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        sizes = tuple(sorted({int(size) for size in pool_sizes}, reverse=True))
        if not sizes or min(sizes) < 1:
            raise ValueError("DAPPM pool sizes must be positive")
        self.pool_sizes = sizes
        self.scale0 = ConvGNAct(in_channels, branch_channels, 1, padding=0)
        self.pool_projections = nn.ModuleList(
            ConvGNAct(in_channels, branch_channels, 1, padding=0)
            for _ in sizes
        )
        self.processes = nn.ModuleList(
            ConvGNAct(branch_channels, branch_channels, 3)
            for _ in sizes
        )
        self.compression = ConvGNAct(
            branch_channels * (len(sizes) + 1),
            out_channels,
            1,
            padding=0,
            activation=False,
        )
        self.shortcut = ConvGNAct(
            in_channels,
            out_channels,
            1,
            padding=0,
            activation=False,
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        output_size = x.shape[-2:]
        previous = self.scale0(x)
        outputs = [previous]
        for configured_size, projection, process in zip(
            self.pool_sizes, self.pool_projections, self.processes
        ):
            grid = max(1, min(configured_size, *output_size))
            pooled = F.adaptive_avg_pool2d(x, (grid, grid))
            pooled = projection(pooled)
            pooled = F.interpolate(
                pooled,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
            previous = process(previous + pooled)
            outputs.append(previous)
        context = self.compression(torch.cat(outputs, dim=1))
        return self.activation(context + self.shortcut(x))


class ControlledRoadFusion(nn.Module):
    """Selectively inject semantic context into the persistent S8 detail path.

    The two branches are normalized independently and concatenated so their
    channel identities are not destroyed by an element-wise sum.  The detail
    stream is the residual anchor; a small learnable per-channel scale lets
    semantic information enter gradually.  One directional RepDepthwise block
    refines the fused road geometry and is deployable as a single DW 5x5 conv.

    When bilateral_fusion="spatial" (already used for the semantic<->detail
    exchange), the same treatment now also applies here: a per-pixel
    ResidualSpatialGate modulates how much of the mixed detail+semantic
    content enters, instead of a single static per-channel scale. This
    closes a gap in the original design -- the exchange gate already had a
    spatial option, this one did not -- reusing the existing, zero-init-safe
    gate module rather than inventing a new mechanism.
    """

    def __init__(
        self,
        channels: int,
        refine_blocks: int = 1,
        deploy: bool = False,
        spatial_gate: bool = False,
    ) -> None:
        super().__init__()
        self.detail_norm = nn.BatchNorm2d(channels)
        self.semantic_norm = nn.BatchNorm2d(channels)
        self.fusion_projection = ConvBNAct(
            channels * 2,
            channels,
            1,
            padding=0,
            activation=False,
        )
        self.fusion_scale = nn.Parameter(
            torch.full((1, channels, 1, 1), 0.10)
        )
        self.spatial_gate = bool(spatial_gate)
        if self.spatial_gate:
            self.fusion_spatial_gate = ResidualSpatialGate(
                channels,
                channels,
                hidden_channels=max(16, min(64, channels // 2)),
            )
        self.refinement = nn.Sequential(
            *[
                RepDepthwiseBlock(channels, deploy=deploy)
                for _ in range(max(1, int(refine_blocks)))
            ]
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, detail: Tensor, semantic: Tensor) -> Tensor:
        if detail.shape[-2:] != semantic.shape[-2:]:
            raise ValueError("Detail and semantic maps must be spatially aligned")
        mixed = self.fusion_projection(
            torch.cat(
                (
                    self.detail_norm(detail),
                    self.semantic_norm(semantic),
                ),
                dim=1,
            )
        )
        if self.spatial_gate:
            mixed = self.fusion_spatial_gate(detail, mixed) * mixed
        fused = self.activation(detail + self.fusion_scale * mixed)
        return self.refinement(fused)


class ResidualSpatialGate(nn.Module):
    """Predict one spatial modulation map for a residual exchange.

    The gate sees independently normalized target and projected-source
    features.  Its output is ``2 * sigmoid(logits)`` rather than a plain
    sigmoid.  Zero-initializing the last convolution therefore starts the
    gate at exactly one, making the spatial variant initially identical to
    the original channel-scaled residual exchange.  Training can then
    suppress clutter and strengthen road-shaped regions without an abrupt
    change to the pretrained feature distribution.
    """

    def __init__(
        self,
        target_channels: int,
        source_channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()
        hidden_channels = max(8, int(hidden_channels))
        self.target_norm = nn.GroupNorm(
            _group_count(target_channels), target_channels
        )
        self.source_norm = nn.GroupNorm(
            _group_count(source_channels), source_channels
        )
        self.mix = nn.Sequential(
            nn.Conv2d(
                target_channels + source_channels,
                hidden_channels,
                3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                _group_count(hidden_channels), hidden_channels
            ),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1, bias=True),
        )
        nn.init.zeros_(self.mix[-1].weight)
        nn.init.zeros_(self.mix[-1].bias)
        self.register_buffer(
            "last_mean", torch.ones((), dtype=torch.float32), persistent=False
        )
        self.register_buffer(
            "last_std", torch.zeros((), dtype=torch.float32), persistent=False
        )

    def forward(self, target: Tensor, projected_source: Tensor) -> Tensor:
        if target.shape[-2:] != projected_source.shape[-2:]:
            raise ValueError("Spatial-gate inputs must be spatially aligned")
        logits = self.mix(
            torch.cat(
                (
                    self.target_norm(target),
                    self.source_norm(projected_source),
                ),
                dim=1,
            )
        )
        gate = 2.0 * torch.sigmoid(logits)
        self.last_mean.copy_(gate.detach().float().mean())
        self.last_std.copy_(gate.detach().float().std(unbiased=False))
        return gate


class DetailGeometryStem(nn.Module):
    """Build the pre-exchange detail representation mainly from S4, not S8.

    The previous detail branch started from ``shared_s8`` -- the exact same
    tensor semantic_s16 is also derived from (both are downstream of
    ResNet's layer2 output). Detail and semantic therefore began as
    near-duplicate views of the same information, which is the root reason
    the detail->semantic exchange had nothing distinctive to contribute:
    logged gate statistics showed that route staying an order of magnitude
    weaker than the reverse direction across multiple full training runs.

    Here the detail stream is rebuilt from the shallower S4 feature
    (ResNet layer1 output) using directional depthwise kernels
    (``RepDepthwiseBlock``'s 3x3 + 1x5 + 5x1 branches) before being
    downsampled to S8 and summed with a light ResNet-S8 projection. The
    result genuinely carries fine road geometry -- width, direction,
    junctions -- that the semantic S16 path, several strides coarser and
    built for context rather than boundaries, does not already encode.
    """

    def __init__(
        self,
        s4_channels: int = 64,
        stem_channels: int = 48,
        s8_channels: int = 128,
        out_channels: int = 96,
        geometry_blocks: int = 2,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.s4_proj = ConvBNAct(s4_channels, stem_channels, 1, padding=0)
        self.geometry = nn.Sequential(
            *[
                RepDepthwiseBlock(stem_channels, deploy=deploy)
                for _ in range(max(1, int(geometry_blocks)))
            ]
        )
        # Depthwise stride-2: cheap S4 -> S8 downsample that keeps the
        # directional geometry learned above intact per-channel, rather
        # than blending channels while also changing resolution.
        self.downsample = nn.Sequential(
            nn.Conv2d(
                stem_channels,
                stem_channels,
                3,
                stride=2,
                padding=1,
                groups=stem_channels,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        self.geometry_to_detail = ConvBNAct(
            stem_channels, out_channels, 1, padding=0, activation=False
        )
        self.resnet_to_detail = ConvBNAct(
            s8_channels, out_channels, 1, padding=0, activation=False
        )
        self.refine = RepDepthwiseBlock(out_channels, deploy=deploy)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, shallow_s4: Tensor, shared_s8: Tensor) -> Tensor:
        geometry = self.geometry(self.s4_proj(shallow_s4))
        geometry_s8 = self.downsample(geometry)
        if geometry_s8.shape[-2:] != shared_s8.shape[-2:]:
            # Odd input sizes can round S4->S8 stride-2 downsampling to a
            # pixel off from ResNet's own S8 stride; align defensively.
            geometry_s8 = F.interpolate(
                geometry_s8,
                size=shared_s8.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fused = self.activation(
            self.geometry_to_detail(geometry_s8)
            + self.resnet_to_detail(shared_s8)
        )
        return self.refine(fused)


class DualResolutionContext(nn.Module):
    """Persistent detail S8 stream plus semantic S16/S32 context stream.

    There is one genuine bilateral interaction at S8 <-> S16, and the two
    directions are deliberately asymmetric rather than mirrored copies of
    each other -- they carry different kinds of information and failed
    differently when built the same way:

    * Semantic -> Detail (context injection, additive residual, unchanged
      from before): semantic tells detail *what* is road under occlusion,
      shadow, or ambiguous texture. This route measured well-used across
      every logged run (scale settling around 0.17-0.33).
    * Detail -> Semantic (geometry modulation, multiplicative gate, NEW):
      detail tells semantic *where exactly* the road boundary/direction
      sits. Previously implemented as an additive residual gated by a
      scale parameter initialized to exact zero
      (``torch.zeros(1, 256, 1, 1)``) -- that zero multiplies away the
      gradient to everything upstream of it (the spatial gate and the
      conv producing the delta) at the very first step, and only recovers
      as the scale itself slowly drifts off zero. Every full run showed
      this route stuck near 0.02-0.07 for 150 epochs as a result. The new
      form is ``semantic' = semantic0 * gate(detail0, semantic0)`` with
      ``gate = 2*sigmoid(zero_init_conv(...))``: gate=1 at init (identity,
      does not disturb pretrained semantics -- same safety property as
      before) but d(gate)/d(conv weights) is nonzero from the first step,
      and the multiplication by the sizeable semantic0 tensor (not a
      near-zero scalar) carries a real gradient signal back into the
      conv immediately. This reuses ``ResidualSpatialGate`` verbatim,
      which already implements exactly this zero-init-safe gate formula.
      The gate is always built regardless of ``bilateral_fusion`` --
      unlike the S->D spatial gate (a style choice between two working
      designs), this is a correctness fix for a dead gradient path, not
      an optional refinement.

    S32 is used only to gather broad DAPPM context; it returns to the
    saved S16 feature as a gated residual before the final S8 fusion. This
    avoids asking the S32 map to preserve thin roads and avoids a second
    heavy bilateral module.

    Two further additions, unrelated to the exchange direction and kept
    from the previous revision:

    * ``strip_pooling`` -- applied to the detail stream right before final
      fusion, giving every pixel full-row/full-column context so a long
      straight road (or a short occluded segment) can be bridged more
      cheaply than deepening the square DAPPM pooling.
    * ``semantic_aux_head`` -- a train-only deep-supervision head reading
      the finished S16 semantic feature (after the S32 DAPPM context has
      been folded back in), giving the semantic branch a direct,
      un-gated reason to become road-discriminative rather than relying
      solely on the (now-fixed) exchange from detail.
    """

    def __init__(
        self,
        detail_channels: int = 96,
        semantic_channels: int = 192,
        dappm_channels: int = 32,
        dappm_pool_sizes: Sequence[int] = (1, 2, 4, 8),
        detail_blocks: Sequence[int] = (2, 2),
        detail_stem_channels: int = 48,
        semantic_blocks: int = 2,
        fusion_blocks: int = 1,
        bilateral_fusion: str = "spatial",
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(detail_blocks) != 2:
            raise ValueError(
                "detail_blocks must contain two depths: "
                "(S4 geometry-stem blocks, post-exchange refine blocks)"
            )
        bilateral_fusion = str(bilateral_fusion).lower()
        if bilateral_fusion not in {"static", "spatial"}:
            raise ValueError(
                "bilateral_fusion must be either 'static' or 'spatial'"
            )
        self.bilateral_fusion = bilateral_fusion

        # Detail is rebuilt from S4 (shallow ResNet layer1 output) rather
        # than sharing shared_s8 with the semantic branch. See
        # DetailGeometryStem's docstring.
        self.detail_stem = DetailGeometryStem(
            s4_channels=64,
            stem_channels=detail_stem_channels,
            s8_channels=128,
            out_channels=detail_channels,
            geometry_blocks=detail_blocks[0],
            deploy=deploy,
        )
        self.detail_refine = nn.Sequential(
            *[
                RepDepthwiseBlock(detail_channels, deploy=deploy)
                for _ in range(max(1, int(detail_blocks[1])))
            ]
        )

        # Pretrained ResNet layer4 now supplies S32 semantics.  A 1x1 adapter
        # replaces the previous randomly initialized stride-2 semantic stage;
        # ``semantic_blocks`` is retained in the public signature so old
        # experiment commands remain valid, but no extra S32 blocks are added.
        _ = semantic_blocks
        # Keep the original checkpoint key name.  The epoch-230 baseline stores
        # these tensors under ``dual_branch.semantic_projection.*``.
        self.semantic_projection = ConvBNAct(
            512, semantic_channels, 1, padding=0
        )

        # Semantic -> Detail: context injection, additive residual.
        self.semantic_to_detail_1 = ConvBNAct(
            256, detail_channels, 1, padding=0, activation=False
        )
        self.semantic_to_detail_scale_1 = nn.Parameter(
            torch.full((1, detail_channels, 1, 1), 0.10)
        )
        if bilateral_fusion == "spatial":
            self.semantic_to_detail_spatial_gate_1 = ResidualSpatialGate(
                detail_channels,
                detail_channels,
                hidden_channels=max(16, min(64, detail_channels // 2)),
            )

        # Detail -> Semantic: geometry modulation, multiplicative gate.
        # Spatial processing (directional refine + stride-2 downsample) is
        # kept separate from channel projection (final 1x1), per the same
        # reasoning as DetailGeometryStem: mixing channels and changing
        # resolution in one conv blurs both jobs.
        self.detail_geometry_to_semantic = nn.Sequential(
            RepDepthwiseBlock(detail_channels, deploy=deploy),
            nn.Conv2d(
                detail_channels,
                detail_channels,
                3,
                stride=2,
                padding=1,
                groups=detail_channels,
                bias=False,
            ),
            nn.BatchNorm2d(detail_channels),
            nn.ReLU(inplace=True),
            ConvBNAct(detail_channels, 256, 1, padding=0, activation=False),
        )
        # Always built: this is the fix for the dead-gradient direction,
        # not a style toggle like the S->D spatial gate above.
        self.detail_to_semantic_gate = ResidualSpatialGate(
            256, 256, hidden_channels=32
        )

        self.dappm = ProgressiveDAPPM(
            semantic_channels,
            dappm_channels,
            semantic_channels,
            pool_sizes=dappm_pool_sizes,
        )
        self.context_to_s16 = ConvBNAct(
            semantic_channels,
            256,
            1,
            padding=0,
            activation=False,
        )
        self.context_scale = nn.Parameter(torch.full((1, 256, 1, 1), 0.10))

        # Train-only deep supervision on the finished S16 semantic feature.
        semantic_aux_hidden = max(32, 256 // 4)
        self.semantic_aux_head = nn.Sequential(
            ConvBNAct(256, semantic_aux_hidden, 3),
            nn.Conv2d(semantic_aux_hidden, 2, 1),
        )

        # Full-row/full-column context for the persistent detail stream.
        self.strip_pooling = StripPoolingModule(detail_channels)

        self.semantic_to_fusion = ConvBNAct(
            256,
            detail_channels,
            1,
            padding=0,
        )
        self.final_fusion = ControlledRoadFusion(
            detail_channels,
            refine_blocks=fusion_blocks,
            deploy=deploy,
            spatial_gate=(bilateral_fusion == "spatial"),
        )
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        shallow_s4: Tensor,
        shared_s8: Tensor,
        semantic_s16: Tensor,
        semantic_s32: Tensor,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        detail0 = self.detail_stem(shallow_s4, shared_s8)
        semantic0 = semantic_s16

        # Semantic -> Detail: context injection.
        semantic_delta = self._resize(
            self.semantic_to_detail_1(semantic0), detail0.shape[-2:]
        )
        if self.bilateral_fusion == "spatial":
            semantic_delta = (
                self.semantic_to_detail_spatial_gate_1(detail0, semantic_delta)
                * semantic_delta
            )
        detail = self.activation(
            detail0 + self.semantic_to_detail_scale_1 * semantic_delta
        )

        # Detail -> Semantic: geometry modulation. No additive residual and
        # no separate scale parameter -- the gate itself (initialized to 1)
        # is the entire mechanism, so there is no zero multiplier for
        # gradients to get stuck behind.
        detail_geometry = self._resize(
            self.detail_geometry_to_semantic(detail0), semantic0.shape[-2:]
        )
        detail_gate = self.detail_to_semantic_gate(semantic0, detail_geometry)
        semantic = semantic0 * detail_gate

        # The detail stream gathers full-row/column road context via strip
        # pooling right before the final gated fusion.
        detail = self.detail_refine(detail)
        detail = self.strip_pooling(detail)

        # S32 gathers context, then returns to the saved S16 representation.
        context_s32 = self.dappm(
            self.semantic_projection(semantic_s32)
        )
        context_s16 = self._resize(
            self.context_to_s16(context_s32), semantic.shape[-2:]
        )
        semantic = self.activation(semantic + self.context_scale * context_s16)

        semantic_aux_logits: Optional[Tensor] = None
        if self.training:
            semantic_aux_logits = self.semantic_aux_head(semantic)

        semantic_s8 = self._resize(
            self.semantic_to_fusion(semantic), detail.shape[-2:]
        )
        fused = self.final_fusion(detail, semantic_s8)
        if self.training:
            return fused, semantic_aux_logits
        return fused

    @torch.no_grad()
    def gate_statistics(self) -> Dict[str, float]:
        """Small diagnostics showing whether each information route is used."""
        gates = {
            "semantic_to_detail": self.semantic_to_detail_scale_1,
            "s32_context_to_s16": self.context_scale,
            "semantic_to_final": self.final_fusion.fusion_scale,
        }
        statistics: Dict[str, float] = {}
        for name, gate in gates.items():
            detached = gate.detach().float()
            statistics[f"{name}_abs_mean"] = float(detached.abs().mean().cpu())
            statistics[f"{name}_abs_max"] = float(detached.abs().max().cpu())
        # detail_to_semantic is now a pure multiplicative gate (identity at
        # init = 1.0, not 0.0), so "how far from 1" is the meaningful
        # activity signal, not "how far from 0".
        statistics["detail_to_semantic_gate_mean"] = float(
            self.detail_to_semantic_gate.last_mean.cpu()
        )
        statistics["detail_to_semantic_gate_std"] = float(
            self.detail_to_semantic_gate.last_std.cpu()
        )
        if self.bilateral_fusion == "spatial":
            statistics["semantic_to_detail_spatial_mean"] = float(
                self.semantic_to_detail_spatial_gate_1.last_mean.cpu()
            )
            statistics["semantic_to_detail_spatial_std"] = float(
                self.semantic_to_detail_spatial_gate_1.last_std.cpu()
            )
        return statistics


class DualBranchRoadNet(nn.Module):
    """Dual-resolution road model with progressive-unfreezing support."""

    PHASE_NAMES = {
        0: "head_only",
        1: "head_plus_dual_branch",
        2: "plus_resnet_layer3_layer4",
        3: "plus_resnet_layer2",
        4: "all_trainable",
    }

    def __init__(
        self,
        num_classes: int = 2,
        detail_channels: int = 96,
        semantic_channels: int = 192,
        dappm_channels: int = 32,
        dappm_pool_sizes: Sequence[int] = (1, 2, 4, 8),
        detail_blocks: Sequence[int] = (2, 2),
        detail_stem_channels: int = 48,
        semantic_blocks: int = 2,
        fusion_blocks: int = 1,
        bilateral_fusion: str = "spatial",
        decoder_s4_channels: int = 64,
        decoder_s2_channels: int = 32,
        full_channels: int = 24,
        dropout: float = 0.05,
        imagenet_pretrained: bool = True,
        encoder_weights_path: Optional[str] = None,
        deploy: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TruncatedResNet34(
            imagenet_pretrained=imagenet_pretrained,
            encoder_weights_path=encoder_weights_path,
        )
        self.dual_branch = DualResolutionContext(
            detail_channels=detail_channels,
            semantic_channels=semantic_channels,
            dappm_channels=dappm_channels,
            dappm_pool_sizes=dappm_pool_sizes,
            detail_blocks=detail_blocks,
            detail_stem_channels=detail_stem_channels,
            semantic_blocks=semantic_blocks,
            fusion_blocks=fusion_blocks,
            bilateral_fusion=bilateral_fusion,
            deploy=deploy,
        )
        self.decode_head = RoadReconstructionDecoder(
            stem_channels=64,
            shallow_channels=64,
            fused_channels=detail_channels,
            s4_channels=decoder_s4_channels,
            s2_channels=decoder_s2_channels,
            full_channels=full_channels,
            num_classes=num_classes,
            dropout=dropout,
            deploy=deploy,
        )
        self.current_phase = 4

    def forward(self, image: Tensor):
        output_size = image.shape[-2:]
        if not self.training or self.current_phase >= 4:
            stem, shallow, shared, semantic, context = self.encoder(image)
        elif self.current_phase <= 1:
            with torch.no_grad():
                stem, shallow, shared, semantic, context = self.encoder(image)
        elif self.current_phase == 2:
            with torch.no_grad():
                stem = self.encoder.stem(image)
                shallow = self.encoder.layer1(self.encoder.maxpool(stem))
                shared = self.encoder.layer2(shallow)
            semantic = self.encoder.layer3(shared)
            context = self.encoder.layer4(semantic)
        else:
            with torch.no_grad():
                stem = self.encoder.stem(image)
                shallow = self.encoder.layer1(self.encoder.maxpool(stem))
            shared = self.encoder.layer2(shallow)
            semantic = self.encoder.layer3(shared)
            context = self.encoder.layer4(semantic)

        if self.training and self.current_phase == 0:
            with torch.no_grad():
                dual_branch_output = self.dual_branch(
                    shallow, shared, semantic, context
                )
        else:
            dual_branch_output = self.dual_branch(
                shallow, shared, semantic, context
            )

        if self.training:
            fused, semantic_aux_logits = dual_branch_output
        else:
            fused = dual_branch_output

        road_logits = self.decode_head(stem, shallow, fused, output_size)
        if self.training:
            return semantic_aux_logits, road_logits
        return road_logits

    def set_trainable_phase(self, phase: int) -> str:
        phase = int(phase)
        if phase not in self.PHASE_NAMES:
            raise ValueError(f"Unknown trainable phase: {phase}")
        self.current_phase = phase
        return self.PHASE_NAMES[phase]

    def enforce_frozen_norm_eval(self, freeze_encoder_bn: bool = True) -> None:
        frozen_modules: list[nn.Module] = []
        if self.current_phase == 0:
            frozen_modules.append(self.dual_branch)
        if self.current_phase <= 1:
            frozen_modules.extend((self.encoder.layer3, self.encoder.layer4))
        if self.current_phase <= 2:
            frozen_modules.append(self.encoder.layer2)
        if self.current_phase <= 3:
            frozen_modules.extend((self.encoder.stem, self.encoder.layer1))
        for frozen in frozen_modules:
            for module in frozen.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        if freeze_encoder_bn:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def trainable_parameter_counts(self) -> Tuple[int, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        modules: list[nn.Module] = [self.decode_head]
        if self.current_phase >= 1:
            modules.append(self.dual_branch)
        if self.current_phase >= 2:
            modules.extend((self.encoder.layer3, self.encoder.layer4))
        if self.current_phase >= 3:
            modules.append(self.encoder.layer2)
        if self.current_phase >= 4:
            modules.extend((self.encoder.stem, self.encoder.layer1))
        trainable = sum(
            parameter.numel()
            for module in modules
            for parameter in module.parameters()
        )
        return trainable, total

    def optimization_modules(self) -> Dict[str, Iterable[Tuple[str, nn.Parameter]]]:
        """Named parameters per optimizer group.

        Names (not just tensors) are needed so build_optimizer() can
        correctly classify broadcast-shaped gate/scale parameters (e.g.
        semantic_to_detail_scale_1, context_scale, fusion_scale -- all
        stored as (1, C, 1, 1) for broadcasting) as "no weight decay", the
        same treatment BatchNorm weight/bias already get. A plain ndim<=1
        check misses these: they are semantically per-channel scales, not
        weight matrices, but their broadcast shape has ndim==4.
        """

        def named(module: nn.Module, prefix: str) -> Iterable[Tuple[str, nn.Parameter]]:
            return (
                (f"{prefix}.{name}", parameter)
                for name, parameter in module.named_parameters()
            )

        return {
            "head": named(self.decode_head, "decode_head"),
            "dual_branch": named(self.dual_branch, "dual_branch"),
            "layer3": itertools.chain(
                named(self.encoder.layer3, "encoder.layer3"),
                named(self.encoder.layer4, "encoder.layer4"),
            ),
            "layer2": named(self.encoder.layer2, "encoder.layer2"),
            "early_encoder": itertools.chain(
                named(self.encoder.stem, "encoder.stem"),
                named(self.encoder.layer1, "encoder.layer1"),
            ),
        }

    def switch_to_deploy(self) -> None:
        for module in list(self.modules()):
            if isinstance(module, (RepVGGBlock, RepDepthwiseBlock)):
                module.switch_to_deploy()


def build_model(args) -> DualBranchRoadNet:
    """Build from an argparse Namespace or compatible attribute container."""
    return DualBranchRoadNet(
        num_classes=2,
        detail_channels=int(args.detail_channels),
        semantic_channels=int(args.semantic_channels),
        dappm_channels=int(args.dappm_channels),
        dappm_pool_sizes=tuple(int(value) for value in args.dappm_pool_sizes),
        detail_blocks=tuple(int(value) for value in args.detail_blocks),
        detail_stem_channels=int(getattr(args, "detail_stem_channels", 48)),
        semantic_blocks=int(args.semantic_blocks),
        fusion_blocks=int(args.fusion_blocks),
        bilateral_fusion=str(getattr(args, "bilateral_fusion", "static")),
        decoder_s4_channels=int(args.decoder_s4_channels),
        decoder_s2_channels=int(args.decoder_s2_channels),
        full_channels=int(args.full_channels),
        dropout=float(args.dropout),
        imagenet_pretrained=bool(args.imagenet_pretrained),
        encoder_weights_path=args.encoder_weights_path,
    )
