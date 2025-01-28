import torch

from pylo.models.Meta_MLP import MetaMLP

model = MetaMLP(39,32,1)

ckpt = torch.load("/home/paulj/projects/lo/ckpt/MuLO_global_step5000_torch.pth")

model_dict = ckpt["torch_params"]

model_dict = {f"network.{k}": v for k, v in model_dict.items()}
model.load_state_dict(model_dict)


model.push_to_hub("Pauljanson002/test")

