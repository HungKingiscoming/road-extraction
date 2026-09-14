"""RoadFusion V4 - Road-Aware Fusion modeling package."""

from .model import (
    TruncatedResNet34,
    ProgressiveDAPPM,
    ControlledRoadFusion,
    ResidualSpatialGate,
    DualResolutionContext,
    DualBranchRoadNet,
    build_model,
)

from .decoder import (
    ConvBNAct,
    ConvGNAct,
    ConvBN,
    RepVGGBlock,
    RepDepthwiseBlock,
    SeparableConvBNAct,
    StripPoolingModule,
    RoadReconstructionDecoder,
    soft_skeletonize,
    binary_dice_loss,
    soft_cldice_loss,
    RoadSegClDiceLoss,
    verify_reparameterization,
)

__all__ = [
    "TruncatedResNet34",
    "ProgressiveDAPPM",
    "ControlledRoadFusion",
    "ResidualSpatialGate",
    "DualResolutionContext",
    "DualBranchRoadNet",
    "build_model",
    "ConvBNAct",
    "ConvGNAct",
    "ConvBN",
    "RepVGGBlock",
    "RepDepthwiseBlock",
    "SeparableConvBNAct",
    "StripPoolingModule",
    "RoadReconstructionDecoder",
    "soft_skeletonize",
    "binary_dice_loss",
    "soft_cldice_loss",
    "RoadSegClDiceLoss",
    "verify_reparameterization",
]
