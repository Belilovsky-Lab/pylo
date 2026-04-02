"""Celo2 MLP model for the learned optimizer.

Architecture: Linear(30, 8) -> ReLU -> Linear(8, 8) -> ReLU -> Linear(8, 3).

The first layer is a single dense matrix applied to the concatenation of all
14 input feature groups (total dim = 30). This is mathematically equivalent to
the original per-group weight approach but simpler and faster.
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
        self.total_input_dim = sum(input_group_dims)

        # First layer: single dense weight applied to concatenated inputs
        self.first_layer = nn.Linear(self.total_input_dim, hidden_size)

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
        # Concatenate all input groups along last dim -> [..., 30]
        x = torch.cat(input_groups, dim=-1)

        # First layer
        out = torch.relu(self.first_layer(x))

        # Hidden layer
        out = torch.relu(self.hidden(out))

        # Output layer
        out = self.output(out)
        return out

    @classmethod
    def from_pretrained_file(cls, path: str) -> "Celo2MLP":
        """Load from a converted .pt checkpoint file.

        Handles both the old per-group format (first_layer_weights.0, etc.)
        and the new dense format (first_layer.weight).
        """
        state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # Convert old per-group format to dense format if needed
        if "first_layer_weights.0" in state_dict:
            state_dict = _convert_grouped_to_dense(state_dict)

        model = cls()
        model.load_state_dict(state_dict)
        return model


def _convert_grouped_to_dense(state_dict: dict) -> OrderedDict:
    """Convert old per-group first_layer_weights.* keys to a single first_layer.*"""
    new_state = OrderedDict()

    # Collect and concatenate the per-group weight matrices
    group_weights = []
    i = 0
    while f"first_layer_weights.{i}" in state_dict:
        group_weights.append(state_dict.pop(f"first_layer_weights.{i}"))
        i += 1

    # Each group weight has shape (hidden_size, group_dim); cat along dim=1
    new_state["first_layer.weight"] = torch.cat(group_weights, dim=1)
    new_state["first_layer.bias"] = state_dict.pop("first_layer_bias")

    # Copy remaining keys unchanged
    new_state.update(state_dict)
    return new_state
