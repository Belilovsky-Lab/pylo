"""Convert a JAX/Haiku CELO2 ``theta`` checkpoint into a PyTorch state_dict.

The original CELO2 meta-parameters are stored as a nested dict
``{"ff_mod_stack": {"<module>": {"w0__0": ..., "b0": ..., "w1": ..., ...}}}``
where every leaf is a ``(in, out)`` weight or a bias. This script flattens that
tree, maps each leaf onto the corresponding :class:`pylo.models.CELO2_MLP.CELO2MLP`
parameter, and saves a ``state_dict`` loadable via
``CELO2_naive(..., checkpoint_path=<out>)``.

Supported input formats (auto-detected by extension / content):
  * ``.pickle`` — a pickled meta-train checkpoint (the framework's format).
  * otherwise   — a flax msgpack checkpoint (requires ``flax`` to be installed).

Run this in an environment that can unpickle the checkpoint (e.g. the
``scaling_l2o`` JAX environment); only ``numpy`` and ``torch`` are required for
the conversion itself.

Usage:
    python scripts/convert_celo2_checkpoint.py \
        --input  /path/to/global_step200000.pickle \
        --output /path/to/celo2_theta.pt
"""

import argparse
import pickle

import numpy as np
import torch

from pylo.models.CELO2_MLP import CELO2MLP


def _load_raw(path):
    if str(path).endswith(".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    # Flax msgpack: deserialize against a reference tree.
    import flax  # noqa: local import; only needed for this branch

    with open(path, "rb") as f:
        return flax.serialization.msgpack_restore(f.read())


def _flatten(tree, prefix=""):
    """Yield (leaf_name, array) pairs from a nested dict of arrays."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}/{k}" if prefix else str(k))
    else:
        yield prefix, np.asarray(tree)


def convert(raw, hidden_size=8, hidden_layers=2):
    # Collect leaves keyed by their final component (b0, w0__3, w1, ...).
    leaves = {}
    for name, arr in _flatten(raw):
        leaves[name.split("/")[-1]] = arr

    model = CELO2MLP(hidden_size=hidden_size, hidden_layers=hidden_layers)
    sd = model.state_dict()

    def assign(dst, src_key):
        src = leaves[src_key]
        if tuple(sd[dst].shape) != tuple(src.shape):
            raise ValueError(
                f"shape mismatch for {dst} <- {src_key}: "
                f"{tuple(sd[dst].shape)} vs {tuple(src.shape)}"
            )
        sd[dst] = torch.tensor(src, dtype=torch.float32)

    # Split first-layer weights: w0__{i} -> w0.{i}
    n_split = len(model.w0)
    for i in range(n_split):
        assign(f"w0.{i}", f"w0__{i}")
    assign("b0", "b0")

    # Dense layers: w1, w2, ... -> dense_w.0, dense_w.1, ...; b1, b2 -> dense_b.*
    n_dense = len(model.dense_w)
    for j in range(n_dense):
        assign(f"dense_w.{j}", f"w{j + 1}")
        assign(f"dense_b.{j}", f"b{j + 1}")

    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="Path to the JAX theta checkpoint.")
    ap.add_argument("--output", required=True, help="Path for the PyTorch state_dict (.pt).")
    ap.add_argument("--hidden-size", type=int, default=8)
    ap.add_argument("--hidden-layers", type=int, default=2)
    args = ap.parse_args()

    raw = _load_raw(args.input)
    model = convert(raw, hidden_size=args.hidden_size, hidden_layers=args.hidden_layers)
    torch.save(model.state_dict(), args.output)
    n = sum(p.numel() for p in model.parameters())
    print(f"Saved CELO2MLP state_dict ({n} params) to {args.output}")


if __name__ == "__main__":
    main()
