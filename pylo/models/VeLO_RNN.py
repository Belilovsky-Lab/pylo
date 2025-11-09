"""
VeLO RNN: An implementation of the RNN component from Google's Versatile Learned Optimizer (VeLO) paper.

This module implements the per-tensor RNN component of the VeLO architecture as described
in Google's Versatile Learned Optimizer paper. The RNN processes tensor-specific features
and outputs control vectors and learning rate multipliers to adapt optimization behavior
for each tensor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin


class LSTM(nn.Module):
    """
    Custom LSTM implementation used within the VeLO architecture.
    
    This LSTM implementation is optimized for the VeLO optimizer's requirements
    and differs slightly from PyTorch's standard LSTM. It uses a single linear
    layer to compute all gates and has a +1 bias for the forget gate to improve
    gradient flow.
    
    Attributes:
        linear (nn.Linear): Linear layer that computes all gate values
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
    """
    def __init__(self, input_size, hidden_size):
        """
        Initialize the LSTM module with specified dimensions.

        Args:
            input_size (int): Size of input features
            hidden_size (int): Size of hidden state
        """
        super(LSTM, self).__init__()
        self.linear = nn.Linear(2 * input_size, 4 * hidden_size)

    def forward(self, x, prev_state):
        """
        Forward pass of the LSTM.

        Args:
            x (torch.Tensor): Input tensor
            prev_state (tuple): Previous hidden state and cell state (h_prev, c_prev)

        Returns:
            tuple: Tuple containing:
                - h (torch.Tensor): New hidden state
                - (h, c) (tuple): New state tuple for next iteration
        """
        h_prev, c_prev = prev_state
        combined = torch.cat((x, h_prev), dim=-1)
        gates = self.linear(combined)
        i, g, f, o = gates.chunk(4, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + 1)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, (h, c)


class VeLORNN(nn.Module, PyTorchModelHubMixin,
    license="apache-2.0", tags=["learned-optimizer"]):
    """
    VeLO RNN module that processes tensor-specific features as described in Google's Versatile Learned Optimizer paper.

    This module implements the per-tensor RNN component of the VeLO architecture. It processes
    tensor-specific features and outputs control vectors and learning rate multipliers that
    determine how parameters are adapted during optimization.

    The VeLORNN applies a feature mixing network (when enabled) followed by an LSTM to
    produce control vectors that weight different parameter initializations and learning
    rate multipliers that scale step sizes.
    """
    def __init__(self, input_size=30, lstm_hidden_size=512, param_inits=256, mix_layers=True):
        """
        Initialize the VeLORNN module.

        Args:
            input_size (int, optional): Dimension of input features. Defaults to 30.
            lstm_hidden_size (int, optional): Size of LSTM hidden state. Defaults to 512.
            param_inits (int, optional): Number of parameter initializations to control.
                Determines the dimension of the output control vector. Defaults to 256.
            mix_layers (bool, optional): Whether to use feature mixing layers before LSTM.
                Defaults to True.
        """
        super(VeLORNN, self).__init__()
        self.mix_layers = mix_layers
        self.mix_layer1 = nn.Linear(input_size, lstm_hidden_size)
        self.mix_layer2 = nn.Linear(input_size, lstm_hidden_size)
        self.final_mix_layer = nn.Linear(input_size, lstm_hidden_size)

        self.lstm = LSTM(input_size=lstm_hidden_size, hidden_size=lstm_hidden_size)
        self.rnn_to_controls = nn.Linear(lstm_hidden_size, param_inits)
        self.step_size = nn.Linear(lstm_hidden_size, 1)
        self.lstm_init_state = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1, lstm_hidden_size)),
                nn.Parameter(torch.zeros(1, lstm_hidden_size)),
            ]
        )

    def forward(self, x, state):
        """
        Forward pass of the VeLORNN.

        This method processes tensor-specific features through optional mixing layers
        and an LSTM to produce control vectors and learning rate multipliers.

        Args:
            x (torch.Tensor): Input tensor containing tensor-specific features
            state (tuple): Previous LSTM state (h, c)

        Returns:
            tuple: Tuple containing:
                - controls (torch.Tensor): Control vector for weighting parameter initializations
                - lr_mult (torch.Tensor): Learning rate multiplier for scaling step size
                - state (tuple): Updated LSTM state for next iteration
        """
        if self.mix_layers:
            # mix_1 = F.relu(self.mix_layer1(x)) #This line is skipped in the original implementation
            mix_2 = F.relu(self.mix_layer2(x))
            v, _ = torch.max(mix_2, dim=0, keepdim=True)
            x = self.final_mix_layer(x) + v
        rnn_out, state = self.lstm(x, state)
        controls = self.rnn_to_controls(rnn_out)
        lr_mult = torch.squeeze(self.step_size(rnn_out), -1)
        return controls, lr_mult, state