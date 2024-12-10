import torch

from pylo.models.Meta_MLP import MetaMLP

model = MetaMLP(39,32,1)

model.load_state_dict(torch.load("/speed-scratch/p_janso/workspace/pylo/test.pt"))

breakpoint()