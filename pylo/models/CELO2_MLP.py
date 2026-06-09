"""CELO2MLP: the per-parameter MLP backbone of the CELO2 learned optimizer.

This mirrors the Haiku ``_ff_mod`` network from the original JAX/optax CELO2
implementation (https://arxiv.org/abs/2602.19142). The defining feature of the
network is a *split* first layer: every input feature group owns its own weight
matrix ``(feature_dim, hidden_size)`` and the first hidden pre-activation is the
sum of ``feature @ weight`` over all groups. The remaining layers are dense.

The forward pass multiplies as ``x @ w`` (Haiku/optax convention, where the
weight is stored ``(in, out)``) rather than ``x @ w.T`` (``torch.nn.Linear``),
so that a checkpoint converted from the JAX implementation reproduces the JAX
output bit-for-bit.
"""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

# Per-feature input dimensions, in the exact order CELO2 builds its features for
# a 2D+ (factored) parameter:
#   g, clip(g), p, m[3], rms[1], m*rsqrt[3], rsqrt[1], fac_g[3], g*rsqrt[1],
#   row_feat[3], col_feat[3], rsqrt(row)[3], rsqrt(col)[3], fac_mom_mult[3]
# A 1D parameter only produces the first 9 groups (no factored features); the
# trailing weights simply go unused for those parameters.
CELO2_FEATURE_DIMS = (1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3)


class CELO2MLP(
    nn.Module,
    PyTorchModelHubMixin,
    license="apache-2.0",
    tags=["learned-optimizer"],
):
    """Split-input MLP used by the CELO2 / ELO-CELO2 optimizers.

    Args:
        feature_dims: Input width of each split first-layer weight. Defaults to
            the 14 feature groups produced by a 2D parameter.
        hidden_size: Width of the hidden layers.
        hidden_layers: Number of hidden weight applications. The architecture is
            ``[hidden_size] * hidden_layers + [output_size]`` (matching the
            original ``[ff_hidden_size] * ff_hidden_layers + [3]``), so the total
            number of weight matrices is ``hidden_layers + 1`` (one split input
            layer plus ``hidden_layers`` dense layers).
        output_size: Width of the final output (CELO2 uses 3; only the first two
            channels, direction and magnitude, are consumed by the optimizer).
        activation: ``"relu"`` or ``"tanh"``.
    """

    def __init__(
        self,
        feature_dims=CELO2_FEATURE_DIMS,
        hidden_size=8,
        hidden_layers=2,
        output_size=3,
        activation="relu",
    ):
        super().__init__()
        self.feature_dims = tuple(feature_dims)
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.activation = activation

        if activation == "relu":
            self.act_fn = torch.relu
        elif activation == "tanh":
            self.act_fn = torch.tanh
        else:
            raise ValueError(f"Invalid MLP activation: {activation}")

        # Split first layer: one (feature_dim, hidden_size) weight per feature.
        self.w0 = nn.ParameterList(
            [nn.Parameter(torch.empty(fd, hidden_size)) for fd in self.feature_dims]
        )
        self.b0 = nn.Parameter(torch.zeros(hidden_size))

        # Dense layers: sizes [hidden_size] * hidden_layers + [output_size].
        layer_sizes = [hidden_size] * hidden_layers + [output_size]
        self.dense_w = nn.ParameterList()
        self.dense_b = nn.ParameterList()
        last = hidden_size
        for size in layer_sizes[1:]:
            self.dense_w.append(nn.Parameter(torch.empty(last, size)))
            self.dense_b.append(nn.Parameter(torch.zeros(size)))
            last = size

        self.reset_parameters()

    def reset_parameters(self):
        # Truncated-normal init mirroring the Haiku stddev = 1 / sqrt(fan_in).
        total_in = sum(self.feature_dims)
        for w in self.w0:
            nn.init.trunc_normal_(w, std=1.0 / (total_in ** 0.5))
        last = self.hidden_size
        for w in self.dense_w:
            nn.init.trunc_normal_(w, std=1.0 / (last ** 0.5))
            last = w.shape[-1]

    def forward(self, inps):
        """Run the split-input MLP.

        Args:
            inps: List of per-feature tensors, each of shape ``[..., feature_dim]``
                and already normalized by the caller. Its length may be shorter
                than ``feature_dims`` (e.g. 9 groups for a 1D parameter), in which
                case only the leading split weights are used.

        Returns:
            Output tensor of shape ``[..., output_size]``.
        """
        o = inps[0] @ self.w0[0]
        for x, w in zip(inps[1:], self.w0[1:]):
            o = o + x @ w
        o = o + self.b0
        o = self.act_fn(o)

        n = len(self.dense_w)
        for i, (w, b) in enumerate(zip(self.dense_w, self.dense_b)):
            o = o @ w + b
            if i != n - 1:
                o = self.act_fn(o)
        return o
