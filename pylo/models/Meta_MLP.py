from collections import OrderedDict
import torch
from huggingface_hub import PyTorchModelHubMixin


class MetaMLP(
    torch.nn.Module,
    PyTorchModelHubMixin,
    license="apache-2.0",
    tags=["learned-optimizer"],
):
    """A Multi-Layer Perceptron model used for meta-learning.

    This MLP architecture is designed specifically for learned optimizers,
    with configurable input size, hidden layer size, and number of hidden layers.
    This follows the architecture described for small_fc_mlp_lopt in the paper Practical Tradeoffs between memory,compute and performance in learned optimizers
    The model implements PyTorch's Module interface and can be pushed to or loaded
    from the Hugging Face Hub.
    """

    def __init__(self, input_size, hidden_size, hidden_layers):
        """Initialize the MetaMLP model.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layers.
            hidden_layers (int): The number of hidden layers in the network.
        """
        super(MetaMLP, self).__init__()
        self.network = torch.nn.Sequential(
            OrderedDict(
                [
                    ("input", torch.nn.Linear(input_size, hidden_size)),
                    ("relu_input", torch.nn.ReLU()),
                ]
            )
        )
        for _ in range(hidden_layers):
            self.network.add_module(
                "linear_{}".format(_), torch.nn.Linear(hidden_size, hidden_size)
            )
            self.network.add_module("relu_{}".format(_), torch.nn.ReLU())
        self.network.add_module("output", torch.nn.Linear(hidden_size, 2))

    def forward(self, x):
        """Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 2].
        """
        return self.network(x)
