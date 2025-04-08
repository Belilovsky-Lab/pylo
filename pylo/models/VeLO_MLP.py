"""
VeLO-MLP: An implementation based on Google's Versatile Learned Optimizer (VeLO) paper.

Some of the following code is adapted from https://github.com/google/learned_optimization/blob/main/learned_optimization/research/general_lopt/hyper_v2.py

This module implements the VeLO-MLP architecture, a neural network model that
serves as a learned optimizer as described in the VeLO paper from Google Research.
The model maintains two sets of parameters:
1. Storage parameters that hold a collection of possible parameter values
2. Actual parameters that are used during forward computation

The model can update its actual parameters based on a control vector through
the update_params method, following the versatile parameter adaptation approach
introduced in the VeLO paper.
"""

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class VeLOMLP(
    nn.Module, PyTorchModelHubMixin, license="apache-2.0", tags=["learned-optimizer"]
):
    """
    Versatile Learned Optimizer MLP (VeLO-MLP).

    This class implements a multi-layer perceptron based on Google's VeLO paper,
    which can adapt its parameters based on a control vector. It maintains two sets of parameters:
    - Storage parameters (with underscore suffix) that maintain a collection of possible parameter values
    - Actual parameters that are used during forward computation

    The model is designed to be used as a learned optimizer where parameters
    can be dynamically updated based on optimization context, implementing the
    versatile parameter adaptation approach described in the VeLO paper.

    """

    def __init__(
        self,
        param_inits=256,
        input_size=30,
        hidden_size=4,
        hidden_layers=1,
        output_size=3,
    ):
        """
        Initialize the VeLOMLP model.

        Args:
            param_inits (int, optional): Number of parameter initializations to maintain in storage.
                Defaults to 256.
            input_size (int, optional): Size of the input dimension. Defaults to 30.
            hidden_size (int, optional): Size of the hidden dimensions. Defaults to 4.
            hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
            output_size (int, optional): Size of the output dimension. Defaults to 3.
        """
        super(VeLOMLP, self).__init__()
        self.hidden_layers = hidden_layers
        # This is to build an architecture to store all the params
        self.input_weights_ = nn.Parameter(
            torch.randn(param_inits, hidden_size, input_size)
        )
        self.input_bias_ = nn.Parameter(torch.zeros(param_inits, hidden_size))
        self.hidden_weights_ = nn.ParameterList()
        self.hidden_bias_ = nn.ParameterList()
        for _ in range(hidden_layers):
            weight = nn.Parameter(torch.randn(param_inits, hidden_size, hidden_size))
            bias = nn.Parameter(torch.zeros(param_inits, hidden_size))
            self.hidden_weights_.append(weight)
            self.hidden_bias_.append(bias)
        self.output_weights_ = nn.Parameter(
            torch.randn(param_inits, output_size, hidden_size)
        )
        self.output_bias_ = nn.Parameter(torch.zeros(param_inits, output_size))

        # This is to define the VeLO-MLP
        self.input_weights = nn.Parameter(torch.randn(hidden_size, input_size))
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.hidden_weights = nn.ParameterList()
        self.hidden_bias = nn.ParameterList()
        for _ in range(hidden_layers):
            weight = nn.Parameter(torch.randn(hidden_size, hidden_size))
            bias = nn.Parameter(torch.zeros(hidden_size))
            self.hidden_weights.append(weight)
            self.hidden_bias.append(bias)
        self.output_weights = nn.Parameter(torch.randn(output_size, hidden_size))
        self.output_bias = nn.Parameter(torch.zeros(output_size))

    def update_params(self, control):
        """
        Update the actual parameters based on the control vector.

        This method computes a weighted average of the storage parameters based on
        the control vector, and updates the actual parameters with the result.
        The weighted average is scaled by a factor of 100.0.

        Args:
            control (torch.Tensor): Control vector that determines the weights for
                parameter averaging. Shape should be compatible with the first
                dimension of storage parameters.

        Returns:
            None
        """
        control_w = control[:, None, None]
        control_b = control[:, None]
        self.input_weights.data.copy_(
            (control_w * self.input_weights_.data).mean(0) * 100.0
        )
        self.input_bias.data.copy_((control_b * self.input_bias_.data).mean(0) * 100.0)
        for i in range(self.hidden_layers):
            self.hidden_weights[i].data.copy_(
                (control_w * self.hidden_weights_[i].data).mean(0) * 100.0
            )
            self.hidden_bias[i].data.copy_(
                (control_b * self.hidden_bias_[i].data).mean(0) * 100.0
            )
        self.output_weights.data.copy_(
            (control_w * self.output_weights_.data).mean(0) * 100.0
        )
        self.output_bias.data.copy_(
            (control_b * self.output_bias_.data).mean(0) * 100.0
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output of the network
        """
        x = F.relu(F.linear(x, self.input_weights, self.input_bias))
        for weight, bias in zip(self.hidden_weights, self.hidden_bias):
            x = F.relu(F.linear(x, weight, bias))
        x = F.linear(x, self.output_weights, self.output_bias)
        return x
