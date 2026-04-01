"""Celo2 MLP model for the learned optimizer.

Architecture: 14 per-input-group first-layer weights -> 8 hidden (ReLU) -> 8 hidden (ReLU) -> 3 output.

The first layer uses separate weight matrices for each input feature group,
whose outputs are summed before adding a shared bias and applying ReLU.
This matches the Haiku-based JAX implementation in celo2_optax.py.
"""

import os
from collections import OrderedDict

import torch
import torch.nn as nn


# Input group dimensions for the default CELO2 config (factored/2D case):
# g(1), clip_g(1), p(1), m(3), rms(1), m*rsqrt(3), rsqrt(1), fac_g(3),
# g*rsqrt(1), row_feat(3), col_feat(3), rsqrt_row(3), rsqrt_col(3), fac_mom_mult(3)
DEFAULT_INPUT_GROUP_DIMS = [1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 3, 3, 3]
DEFAULT_HIDDEN_SIZE = 8
DEFAULT_OUTPUT_SIZE = 3


class Celo2MLP(nn.Module):
    def __init__(
        self,
        input_group_dims=None,
        hidden_size=DEFAULT_HIDDEN_SIZE,
        output_size=DEFAULT_OUTPUT_SIZE,
    ):
        super().__init__()
        if input_group_dims is None:
            input_group_dims = DEFAULT_INPUT_GROUP_DIMS

        self.input_group_dims = input_group_dims
        self.hidden_size = hidden_size
        self.num_groups = len(input_group_dims)

        # First layer: separate weight matrix per input group, shared bias
        self.first_layer_weights = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_size, dim)) for dim in input_group_dims]
        )
        self.first_layer_bias = nn.Parameter(torch.zeros(hidden_size))

        # Hidden layer
        self.hidden = nn.Linear(hidden_size, hidden_size)

        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, input_groups: list[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            input_groups: List of 14 tensors, each with shape [..., group_dim].

        Returns:
            Tensor of shape [..., 3] (direction, magnitude, unused).
        """
        # First layer: sum of per-group matmuls + shared bias
        out = torch.zeros(
            input_groups[0].shape[:-1] + (self.hidden_size,),
            device=input_groups[0].device,
            dtype=input_groups[0].dtype,
        )
        for inp, w in zip(input_groups, self.first_layer_weights):
            out = out + inp @ w.t()
        out = torch.relu(out + self.first_layer_bias)

        # Hidden layer
        out = torch.relu(self.hidden(out))

        # Output layer
        out = self.output(out)
        return out

    @classmethod
    def from_pretrained_file(cls, path: str) -> "Celo2MLP":
        """Load from a converted .pt checkpoint file."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = cls()
        model.load_state_dict(state_dict)
        return model
