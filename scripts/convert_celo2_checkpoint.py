"""Convert CELO2 JAX/Flax checkpoint (theta.state) to PyTorch format.

JAX linear layers use x @ W (W shape: in, out).
PyTorch nn.Linear uses x @ W.T (W shape: out, in).
So all weight matrices are transposed during conversion.

Usage:
    python scripts/convert_celo2_checkpoint.py \
        --input snippets/celo2/celo2/theta.state \
        --output pylo/models/celo2_weights.pt
"""

import argparse
import sys
from collections import OrderedDict

import torch


def convert_checkpoint(input_path: str, output_path: str):
    # Import JAX/Flax only for conversion
    try:
        import flax.serialization
        import jax
        import jax.numpy as jnp
    except ImportError:
        print("ERROR: flax, jax required for conversion. pip install flax jax jaxlib")
        sys.exit(1)

    sys.path.insert(0, "snippets/celo2")
    from celo2_optax import Celo2Transformation

    # Load with structure reference
    ref = Celo2Transformation().init_meta_params(jax.random.PRNGKey(0))
    with open(input_path, "rb") as f:
        theta = flax.serialization.from_bytes(ref, f.read())

    params = theta["ff_mod_stack"]["~"]

    state_dict = OrderedDict()

    # First layer: 14 per-input-group weight matrices
    # JAX shape (in_dim, 8) -> PyTorch shape (8, in_dim)
    for i in range(14):
        jax_w = params[f"w0__{i}"]
        state_dict[f"first_layer_weights.{i}"] = torch.tensor(
            jax_w.T.copy(), dtype=torch.float32
        )

    # First layer bias (shared)
    state_dict["first_layer_bias"] = torch.tensor(
        params["b0"].copy(), dtype=torch.float32
    )

    # Hidden layer 1: (8, 8) -> (8, 8)
    state_dict["hidden.weight"] = torch.tensor(
        params["w1"].T.copy(), dtype=torch.float32
    )
    state_dict["hidden.bias"] = torch.tensor(
        params["b1"].copy(), dtype=torch.float32
    )

    # Output layer: (8, 3) -> (3, 8)
    state_dict["output.weight"] = torch.tensor(
        params["w2"].T.copy(), dtype=torch.float32
    )
    state_dict["output.bias"] = torch.tensor(
        params["b2"].copy(), dtype=torch.float32
    )

    torch.save(state_dict, output_path)
    print(f"Converted checkpoint saved to {output_path}")
    print("Keys and shapes:")
    for k, v in state_dict.items():
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="snippets/celo2/celo2/theta.state")
    parser.add_argument("--output", default="pylo/models/celo2_weights.pt")
    args = parser.parse_args()
    convert_checkpoint(args.input, args.output)
