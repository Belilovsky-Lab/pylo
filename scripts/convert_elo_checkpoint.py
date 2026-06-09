"""Convert a JAX/Haiku ELO (Adafactor-MLP) ``theta`` checkpoint to a PyTorch
:class:`pylo.models.Meta_MLP.MetaMLP` state_dict.

The ELO meta-parameters are stored as
``{"nn": {"<module>": {"w0": (39, 32), "b0": (32,), "w1": (32, 32), "b1": ...,
"w2": (32, 2), "b2": (2,)}}}`` — a standard dense MLP. Haiku stores linear
weights as ``(in, out)`` and applies ``x @ w``; ``torch.nn.Linear`` stores
``(out, in)`` and applies ``x @ w.T``, so each weight is transposed.

Run this in an environment that can unpickle the checkpoint (e.g. the
``scaling_l2o`` JAX environment); only ``numpy`` and ``torch`` are required.

Usage:
    python scripts/convert_elo_checkpoint.py \
        --input  /path/to/global_step100000.pickle \
        --output /path/to/elo_theta.pt
"""

import argparse
import pickle

import numpy as np
import torch

from pylo.models.Meta_MLP import MetaMLP


def _load_raw(path):
    if str(path).endswith(".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    import flax  # noqa: local import; only needed for the msgpack branch

    with open(path, "rb") as f:
        return flax.serialization.msgpack_restore(f.read())


def _flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flatten(v, f"{prefix}/{k}" if prefix else str(k))
    else:
        yield prefix, np.asarray(tree)


def convert(raw, input_size=39, hidden_size=32, hidden_layers=1):
    leaves = {name.split("/")[-1]: arr for name, arr in _flatten(raw)}

    model = MetaMLP(
        input_size=input_size, hidden_size=hidden_size, hidden_layers=hidden_layers
    )
    sd = model.state_dict()

    # MetaMLP dense layers in forward order: input, linear_0..linear_{n-1}, output.
    targets = ["network.input"] + [
        f"network.linear_{i}" for i in range(hidden_layers)
    ] + ["network.output"]

    for j, dst in enumerate(targets):
        w = leaves[f"w{j}"]  # (in, out) -> nn.Linear (out, in)
        b = leaves[f"b{j}"]
        sd[f"{dst}.weight"] = torch.tensor(w, dtype=torch.float32).t().contiguous()
        sd[f"{dst}.bias"] = torch.tensor(b, dtype=torch.float32)

    model.load_state_dict(sd)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--input-size", type=int, default=39)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--hidden-layers", type=int, default=1)
    args = ap.parse_args()

    raw = _load_raw(args.input)
    model = convert(
        raw,
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
    )
    torch.save(model.state_dict(), args.output)
    n = sum(p.numel() for p in model.parameters())
    print(f"Saved MetaMLP state_dict ({n} params) to {args.output}")


if __name__ == "__main__":
    main()
