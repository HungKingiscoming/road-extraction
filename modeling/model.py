from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .decoder import (
    ConvBNAct,
    ConvGNAct,
    DeformableConvBlock,
    DeformableRoadRefineBlock,
    DilatedStripBlock,
    RoadReconstructionDecoder,
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


def _detail_stage(channels: int, blocks: int, deploy: bool) -> nn.Sequential:
    return nn.Sequential(
        *[
            DeformableConvBlock(channels, channels, deploy=deploy)
            for _ in range(max(1, int(blocks)))
        ]
    )


def _semantic_stage(channels: int, blocks: int, dilation: int) -> nn.Sequential:
    return nn.Sequential(
        *[
            DilatedStripBlock(channels, dilation=dilation)
            for _ in range(max(1, int(blocks)))
        ]
    )


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
    """

    def __init__(
        self,
        channels: int,
        refine_blocks: int = 1,
        deploy: bool = False,
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
        self.refinement = nn.Sequential(
            *[
                DeformableRoadRefineBlock(channels, deploy=deploy)
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
        fused = self.activation(detail + self.fusion_scale * mixed)
        return self.refinement(fused)


def _group_count(channels: int, maximum: int = 8) -> int:
    """Largest small GroupNorm divisor, robust for small per-GPU batches."""
    for groups in range(min(maximum, int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


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


class DualResolutionContext(nn.Module):
    """Detail S8 stream plus semantic S16/S32 context stream.

    Both branches now have two stages, exchanging bilaterally TWICE -- once
    after each branch's first stage, once after each branch's second stage
    -- instead of the original single S8<->S16 exchange. This makes the two
    branches structurally symmetric. S32 still gathers DAPPM context only
    once, after the semantic branch's own stage processing is done, and is
    never given its own bilateral exchange -- unchanged from the original
    design rationale: S32 should not be asked to preserve thin roads.
    """

    def __init__(
        self,
        detail_channels: int = 96,
        semantic_channels: int = 192,
        dappm_channels: int = 32,
        dappm_pool_sizes: Sequence[int] = (1, 2, 4, 8),
        detail_blocks: Sequence[int] = (2, 2),
        semantic_blocks: Sequence[int] = (2, 2),
        semantic_dilations: Sequence[int] = (2, 4),
        fusion_blocks: int = 1,
        bilateral_fusion: str = "spatial",
        deploy: bool = False,
    ) -> None:
        super().__init__()
        if len(detail_blocks) != 2:
            raise ValueError("detail_blocks must contain two stage depths")
        if len(semantic_blocks) != 2:
            raise ValueError("semantic_blocks must contain two stage depths")
        if len(semantic_dilations) != 2:
            raise ValueError("semantic_dilations must contain two values")
        bilateral_fusion = str(bilateral_fusion).lower()
        if bilateral_fusion not in {"static", "spatial"}:
            raise ValueError(
                "bilateral_fusion must be either 'static' or 'spatial'"
            )
        self.bilateral_fusion = bilateral_fusion

        # --- Detail branch: persistent S8, two stages (unchanged). ---
        self.detail_projection = ConvBNAct(128, detail_channels, 1, padding=0)
        self.detail_stages = nn.ModuleList(
            _detail_stage(detail_channels, depth, deploy)
            for depth in detail_blocks
        )

        # --- Semantic branch: now has its own two dedicated stages at S16,
        # symmetric with the detail branch, instead of using the raw
        # pretrained layer3 output directly (as the original single-stage
        # design did). DilatedStripBlock keeps a road-shape prior while
        # reaching further per stage via increasing dilation.
        self.semantic_projection_early = ConvBNAct(
            256, semantic_channels, 1, padding=0
        )
        self.semantic_stages = nn.ModuleList(
            _semantic_stage(semantic_channels, depth, dilation)
            for depth, dilation in zip(semantic_blocks, semantic_dilations)
        )

        # --- Bilateral exchange 1: after each branch's first stage. ---
        self.semantic_to_detail_1 = ConvBNAct(
            semantic_channels, detail_channels, 1, padding=0, activation=False
        )
        self.detail_to_semantic_1 = ConvBNAct(
            detail_channels, semantic_channels, 3, stride=2, activation=False
        )
        self.semantic_to_detail_scale_1 = nn.Parameter(
            torch.full((1, detail_channels, 1, 1), 0.10)
        )
        self.detail_to_semantic_scale_1 = nn.Parameter(
            torch.zeros(1, semantic_channels, 1, 1)
        )

        # --- Bilateral exchange 2: after each branch's second stage. ---
        self.semantic_to_detail_2 = ConvBNAct(
            semantic_channels, detail_channels, 1, padding=0, activation=False
        )
        self.detail_to_semantic_2 = ConvBNAct(
            detail_channels, semantic_channels, 3, stride=2, activation=False
        )
        self.semantic_to_detail_scale_2 = nn.Parameter(
            torch.full((1, detail_channels, 1, 1), 0.10)
        )
        self.detail_to_semantic_scale_2 = nn.Parameter(
            torch.zeros(1, semantic_channels, 1, 1)
        )

        if bilateral_fusion == "spatial":
            # Single-channel spatial gates are intentionally used instead of
            # C-channel attention maps -- same rationale as the original
            # design, now duplicated for both exchange points.
            gate_hidden = max(16, min(64, detail_channels // 2))
            self.semantic_to_detail_spatial_gate_1 = ResidualSpatialGate(
                detail_channels, detail_channels, hidden_channels=gate_hidden
            )
            self.detail_to_semantic_spatial_gate_1 = ResidualSpatialGate(
                semantic_channels, semantic_channels, hidden_channels=32
            )
            self.semantic_to_detail_spatial_gate_2 = ResidualSpatialGate(
                detail_channels, detail_channels, hidden_channels=gate_hidden
            )
            self.detail_to_semantic_spatial_gate_2 = ResidualSpatialGate(
                semantic_channels, semantic_channels, hidden_channels=32
            )

        # --- S32 -> DAPPM -> back into the semantic S16 stream. Gathered
        # only once, after the semantic branch has finished both of its own
        # stages and both bilateral exchanges -- unchanged in spirit from
        # the original design.
        self.semantic_projection = ConvBNAct(
            512, semantic_channels, 1, padding=0
        )
        self.dappm = ProgressiveDAPPM(
            semantic_channels,
            dappm_channels,
            semantic_channels,
            pool_sizes=dappm_pool_sizes,
        )
        self.context_to_s16 = ConvBNAct(
            semantic_channels,
            semantic_channels,
            1,
            padding=0,
            activation=False,
        )
        self.context_scale = nn.Parameter(
            torch.full((1, semantic_channels, 1, 1), 0.10)
        )
        self.semantic_to_fusion = ConvBNAct(
            semantic_channels,
            detail_channels,
            1,
            padding=0,
        )
        self.final_fusion = ControlledRoadFusion(
            detail_channels,
            refine_blocks=fusion_blocks,
            deploy=deploy,
        )
        self.activation = nn.ReLU(inplace=True)

    @staticmethod
    def _resize(x: Tensor, size: Tuple[int, int]) -> Tensor:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def _exchange(
        self,
        detail: Tensor,
        semantic: Tensor,
        semantic_to_detail: nn.Module,
        detail_to_semantic: nn.Module,
        semantic_to_detail_scale: Tensor,
        detail_to_semantic_scale: Tensor,
        semantic_to_detail_gate: Optional[nn.Module],
        detail_to_semantic_gate: Optional[nn.Module],
    ) -> Tuple[Tensor, Tensor]:
        detail_before, semantic_before = detail, semantic
        semantic_delta = self._resize(
            semantic_to_detail(semantic_before), detail_before.shape[-2:]
        )
        detail_delta = self._resize(
            detail_to_semantic(detail_before), semantic_before.shape[-2:]
        )
        if self.bilateral_fusion == "spatial":
            semantic_delta = (
                semantic_to_detail_gate(detail_before, semantic_delta)
                * semantic_delta
            )
            detail_delta = (
                detail_to_semantic_gate(semantic_before, detail_delta)
                * detail_delta
            )
        detail = self.activation(
            detail_before + semantic_to_detail_scale * semantic_delta
        )
        semantic = self.activation(
            semantic_before + detail_to_semantic_scale * detail_delta
        )
        return detail, semantic

    def forward(
        self,
        shared_s8: Tensor,
        semantic_s16: Tensor,
        semantic_s32: Tensor,
    ) -> Tensor:
        detail = self.detail_stages[0](self.detail_projection(shared_s8))
        semantic = self.semantic_stages[0](
            self.semantic_projection_early(semantic_s16)
        )

        detail, semantic = self._exchange(
            detail,
            semantic,
            self.semantic_to_detail_1,
            self.detail_to_semantic_1,
            self.semantic_to_detail_scale_1,
            self.detail_to_semantic_scale_1,
            getattr(self, "semantic_to_detail_spatial_gate_1", None),
            getattr(self, "detail_to_semantic_spatial_gate_1", None),
        )

        detail = self.detail_stages[1](detail)
        semantic = self.semantic_stages[1](semantic)

        detail, semantic = self._exchange(
            detail,
            semantic,
            self.semantic_to_detail_2,
            self.detail_to_semantic_2,
            self.semantic_to_detail_scale_2,
            self.detail_to_semantic_scale_2,
            getattr(self, "semantic_to_detail_spatial_gate_2", None),
            getattr(self, "detail_to_semantic_spatial_gate_2", None),
        )

        # S32 gathers context once, then returns to the S16 representation.
        context_s32 = self.dappm(self.semantic_projection(semantic_s32))
        context_s16 = self._resize(
            self.context_to_s16(context_s32), semantic.shape[-2:]
        )
        semantic = self.activation(semantic + self.context_scale * context_s16)

        semantic_s8 = self._resize(
            self.semantic_to_fusion(semantic), detail.shape[-2:]
        )
        return self.final_fusion(detail, semantic_s8)

    @torch.no_grad()
    def gate_statistics(self) -> Dict[str, float]:
        """Small diagnostics showing whether each information route is used."""
        gates = {
            "semantic_to_detail_1": self.semantic_to_detail_scale_1,
            "detail_to_semantic_1": self.detail_to_semantic_scale_1,
            "semantic_to_detail_2": self.semantic_to_detail_scale_2,
            "detail_to_semantic_2": self.detail_to_semantic_scale_2,
            "s32_context_to_s16": self.context_scale,
            "semantic_to_final": self.final_fusion.fusion_scale,
        }
        statistics: Dict[str, float] = {}
        for name, gate in gates.items():
            detached = gate.detach().float()
            statistics[f"{name}_abs_mean"] = float(detached.abs().mean().cpu())
            statistics[f"{name}_abs_max"] = float(detached.abs().max().cpu())
        if self.bilateral_fusion == "spatial":
            for suffix in ("1", "2"):
                for direction in ("semantic_to_detail", "detail_to_semantic"):
                    gate_module = getattr(
                        self, f"{direction}_spatial_gate_{suffix}"
                    )
                    statistics[f"{direction}_spatial_mean_{suffix}"] = float(
                        gate_module.last_mean.cpu()
                    )
                    statistics[f"{direction}_spatial_std_{suffix}"] = float(
                        gate_module.last_std.cpu()
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
        semantic_blocks: Sequence[int] = (2, 2),
        semantic_dilations: Sequence[int] = (2, 4),
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
            semantic_blocks=semantic_blocks,
            semantic_dilations=semantic_dilations,
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
                fused = self.dual_branch(shared, semantic, context)
        else:
            fused = self.dual_branch(shared, semantic, context)
        return self.decode_head(stem, shallow, fused, output_size)

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

    def optimization_modules(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "head": self.decode_head.parameters(),
            "dual_branch": self.dual_branch.parameters(),
            "layer3": (
                parameter
                for module in (self.encoder.layer3, self.encoder.layer4)
                for parameter in module.parameters()
            ),
            "layer2": self.encoder.layer2.parameters(),
            "early_encoder": (
                parameter
                for module in (self.encoder.stem, self.encoder.layer1)
                for parameter in module.parameters()
            ),
        }

    def switch_to_deploy(self) -> None:
        """Không còn tác dụng: kiến trúc hiện tại toàn bộ dùng deformable
        conv (offset phụ thuộc input), không có block nào fuse được về dạng
        tĩnh. Giữ lại phương thức để không phá vỡ code gọi
        model.switch_to_deploy() ở nơi khác (ví dụ script export)."""
        return


def build_model(args) -> DualBranchRoadNet:
    """Build from an argparse Namespace or compatible attribute container."""
    return DualBranchRoadNet(
        num_classes=2,
        detail_channels=int(args.detail_channels),
        semantic_channels=int(args.semantic_channels),
        dappm_channels=int(args.dappm_channels),
        dappm_pool_sizes=tuple(int(value) for value in args.dappm_pool_sizes),
        detail_blocks=tuple(int(value) for value in args.detail_blocks),
        semantic_blocks=tuple(int(value) for value in args.semantic_blocks),
        semantic_dilations=tuple(
            int(value) for value in getattr(args, "semantic_dilations", (2, 4))
        ),
        fusion_blocks=int(args.fusion_blocks),
        bilateral_fusion=str(getattr(args, "bilateral_fusion", "static")),
        decoder_s4_channels=int(args.decoder_s4_channels),
        decoder_s2_channels=int(args.decoder_s2_channels),
        full_channels=int(args.full_channels),
        dropout=float(args.dropout),
        imagenet_pretrained=bool(args.imagenet_pretrained),
        encoder_weights_path=args.encoder_weights_path,
    )
