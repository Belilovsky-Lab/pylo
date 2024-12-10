from collections import OrderedDict
import torch
from huggingface_hub import PyTorchModelHubMixin


class MetaMLP(torch.nn.Module, PyTorchModelHubMixin,repo_url="Pauljanson002/test",
    pipeline_tag="learned-optimizer",
    license="apache-2.0",):
    
    def __init__(self, input_size, hidden_size, hidden_layers):
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
        return self.network(x)