import torch

from pylo.models.Meta_MLP import MetaMLP
from pylo.models.VeLO_MLP import VeLOMLP
from pylo.models.VeLO_RNN import VeLORNN

# model = MetaMLP(39,32,1)

# ckpt = torch.load("/home/paulj/projects/lo/ckpt/MuLO_global_step5000_torch.pth")

# model_dict = ckpt["torch_params"]

# model_dict = {f"network.{k}": v for k, v in model_dict.items()}
# model.load_state_dict(model_dict)


# model.push_to_hub("Pauljanson002/test")

# model = VeLORNN()

# dicts = torch.load("/home/paulj/workspace/pylo/VeLO_torch.pth")
# model.load_state_dict(dicts["rnn_params"], strict=False)
# model.lstm_init_state[0].data.copy_(dicts["lstm_init_state"][0])
# model.lstm_init_state[1].data.copy_(dicts["lstm_init_state"][1])
# model.push_to_hub("Pauljanson002/VeLO_RNN")
# print("Model successfully pushed to hub.")

# model = VeLOMLP()
# dicts = torch.load("/home/paulj/workspace/pylo/VeLO_torch.pth")
# model.load_state_dict(dicts["ff_mod_stack"], strict=False)
# model.push_to_hub("Pauljanson002/VeLO_MLP")
# print("Model successfully pushed to hub.")
