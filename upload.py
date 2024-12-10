import torch

from pylo.models.Meta_MLP import MetaMLP

model = MetaMLP(39,32,1)

model.push_to_hub("Pauljanson002/test")

breakpoint()