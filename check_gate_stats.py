"""Print DualResolutionContext's learned gate scales from a trained checkpoint.

Every cross-branch/cross-resolution path in the model (bilateral S8<->S16
exchange, DAPPM context injected back into S16, final semantic->detail
fusion) is a small, zero/near-zero-initialized residual gate. This prints how
far each has actually grown from its init value after training, which is
direct evidence of whether that pathway learned to contribute meaningfully
or stayed suppressed.
"""

import argparse

import torch

from modeling.model import build_model
from test_native import clean_state_dict, resolve_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to a .pt file or a checkpoint directory")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    saved_args = dict(checkpoint["args"])
    saved_args["imagenet_pretrained"] = False
    saved_args["encoder_weights_path"] = None
    model = build_model(argparse.Namespace(**saved_args))

    state = checkpoint.get(args.weights) or checkpoint.get(
        "model" if args.weights == "ema" else "ema"
    )
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.eval()

    print(f"Checkpoint: {checkpoint_path} (weights={args.weights})")
    print(f"oriented_skip={saved_args.get('oriented_skip')}  "
          f"epoch={checkpoint.get('epoch')}\n")

    # Populate the spatial-gate last_mean/last_std buffers with one forward
    # pass; without this they still hold their register_buffer init values
    # (1.0 / 0.0) and would look identical to "gate never moved" even though
    # it just never ran. Random noise input, so treat spatial numbers only as
    # a rough sanity check -- the four scalar gates below need no forward
    # pass and are the primary signal.
    with torch.no_grad():
        dummy = torch.randn(1, 3, 512, 512)
        model(dummy)

    stats = model.dual_branch.gate_statistics()
    init_values = {
        "semantic_to_detail_abs_mean": 0.10,
        "detail_to_semantic_abs_mean": 0.0,
        "s32_context_to_s16_abs_mean": 0.10,
        "semantic_to_final_abs_mean": 0.10,
    }
    print(f"{'gate':38s}{'value':>10s}{'init':>10s}")
    for name, value in stats.items():
        init = init_values.get(name)
        init_str = f"{init:.4f}" if init is not None else ""
        print(f"{name:38s}{value:10.4f}{init_str:>10s}")
    print(
        "\n(spatial_mean/spatial_std above are from one random-noise forward "
        "pass -- a rough sanity check only, not real-image statistics)"
    )


if __name__ == "__main__":
    main()
