import torch
from pylo.models.Meta_MLP import MetaMLP


model = MetaMLP.from_pretrained("Pauljanson002/test")

offline_model = MetaMLP(39,32,1)
offline_model.load_state_dict(torch.load("/speed-scratch/p_janso/workspace/pylo/test.pt"))

# check that the models are the same
if torch.allclose(model.network[0].weight, offline_model.network[0].weight):
    print("Models are the same")