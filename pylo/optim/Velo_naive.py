"""
VeLO: An implementation of VeLO from https://arxiv.org/abs/2211.09760.

Some of the following code is adapted from https://github.com/google/learned_optimization/blob/main/learned_optimization/research/general_lopt/hyper_v2.py
"""
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Optimizer
import torch.nn.functional as F
from collections import OrderedDict

from pylo.models.VeLO_MLP import VeLOMLP
from pylo.models.VeLO_RNN import VeLORNN

def init_factors(p):
    shape = p.shape
    f_dims = factored_dims(shape)
    shape = shape + (3,)
    if f_dims is not None:
        d1, d0 = f_dims
        vr_shape = tuple(dim for i, dim in enumerate(shape) if i != d0)
        vc_shape = tuple(dim for i, dim in enumerate(shape) if i != d1)
        v_row = torch.zeros(vr_shape, dtype=torch.float32)
        v_col = torch.zeros(vc_shape, dtype=torch.float32)
        return v_row, v_col, torch.tensor([], dtype=torch.float32)

    else:
        v = torch.zeros(shape, dtype=torch.float32)
        return (
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
            v,
        )


def update_factors(
    v_col, v_row, v_full, g, g_shape, decay_rate: float = 0.9, epsilon: float = 1e-30
):
    f_dims = factored_dims(g_shape)
    mixing_rate = 1.0 - decay_rate
    rp_shape = [1] * len(g_shape)
    g = g.repeat(rp_shape + [decay_rate.shape[-1]])
    grad_sqr = g * g + epsilon

    if f_dims is not None:
        d1, d0 = f_dims
        decay_rate, mixing_rate = decay_rate.squeeze(0), mixing_rate.squeeze(0)
        new_v_row = decay_rate * v_row + mixing_rate * torch.mean(grad_sqr, dim=d0)
        new_v_col = decay_rate * v_col + mixing_rate * torch.mean(grad_sqr, dim=d1)

        reduced_d1 = d1 - 1 if d1 > d0 else d1
        row_col_mean = torch.mean(new_v_row, dim=reduced_d1, keepdim=True)

        row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
        col_factor = safe_rsqrt(new_v_col)
        y = g * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
        return new_v_col, new_v_row, torch.tensor([], dtype=torch.float32), y

    else:
        new_v = decay_rate * v_full + mixing_rate * grad_sqr
        y = g * safe_rsqrt(new_v + 1e-9)
        return (
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
            new_v,
            y,
        )


def tanh_embedding(x):
    x = torch.tensor(x, dtype=torch.float32)
    timescales = torch.tensor(
        [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000], dtype=torch.float32
    )
    embeddings = torch.tanh(x / timescales - 1.0)
    return embeddings


def second_moment_normalizer(x, axis, eps=1e-5):
    mean_squared = torch.mean(torch.square(x), dim=axis, keepdim=True)
    return x * torch.rsqrt(eps + mean_squared)


def factored_dims(shape):
    if len(shape) < 2:
        return None
    sorted_dims = np.argsort(shape)
    if shape[int(sorted_dims[-2])] == shape[int(sorted_dims[-1])]:
        if len(shape) == 4 and int(sorted_dims[-2]) == 0 and int(sorted_dims[-1]) == 1:
            return int(sorted_dims[-2]), int(sorted_dims[-1])
        else:
            return int(sorted_dims[-1]), int(sorted_dims[-2])
    else:
        return int(sorted_dims[-2]), int(sorted_dims[-1])


def decay_to_param(x):
    return torch.log(1 - x) / 10.0


def param_to_decay(x):
    return 1 - torch.exp(x * 10.0)


def safe_rsqrt(x):
    return torch.rsqrt(torch.maximum(x, torch.tensor(1e-9)))


def clip_log_abs(v, scale=1.0):
    mag = torch.log(1e-8 + torch.abs(v * scale))
    return torch.clamp(mag, -5, 5) * 0.5


def sorted_values(dd):
    return list(zip(*sorted(dd.items(), key=lambda x: x[0])))[1]


def fractional_tanh_embed(x):
    def one_freq(timescale):
        return torch.tanh((x - timescale) * 10)

    timescales = torch.tensor(
        [0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1], dtype=torch.float32
    )
    return torch.stack([one_freq(ts) for ts in timescales])


class BufferLossAccumulators:
    def __init__(self, device):
        self.device = device
        pass

    def init(self, num_steps):
        halflife = torch.logspace(
            1, torch.log10(torch.tensor(num_steps, dtype=torch.float32)), 10
        )
        decays = torch.exp(-1.0 / halflife)
        return {
            "means": torch.zeros(len(decays), dtype=torch.float32, device=self.device),
            "iteration": torch.tensor(0, dtype=torch.int32, device=self.device),
            "running_min": 999999999999.0
            * torch.ones(len(decays), dtype=torch.float32, device=self.device),
            "decays": decays.to(self.device),
        }

    def update(self, state, loss):
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** (state["iteration"] + 1))
        approx_max = torch.max(cor_mean)
        approx_max = torch.where(state["iteration"] == 0, loss, approx_max)
        loss = torch.minimum(torch.abs(approx_max) * 2, loss)
        means = state["means"] * jdecays + loss * (1.0 - jdecays)
        cor_mean = means / (1 - jdecays ** (state["iteration"] + 1))
        running_min = torch.minimum(state["running_min"], cor_mean)

        return {
            "means": means,
            "iteration": state["iteration"] + 1,
            "running_min": running_min,
            "decays": state["decays"],
        }

    def features(self, state):
        jdecays = state["decays"]
        cor_mean = state["means"] / (1 - jdecays ** state["iteration"])
        approx_max = cor_mean[1:]
        cor_mean = cor_mean[0:-1]
        running_min = state["running_min"][0:-1]

        den = torch.maximum(torch.tensor(1e-8), (approx_max - running_min))
        pre_center = (cor_mean - running_min) / den
        feature1 = pre_center - 1.0
        feature1 = torch.clamp(feature1, -1, 1)
        return torch.where(state["iteration"] <= 2, feature1 * 0, feature1)


def lstm_features_for_tensor(p, g, m, rms, fraction_trained, loss_features, device):

    norm_mult = torch.rsqrt(torch.clamp(torch.mean(p**2), min=1e-9))
    g = g * norm_mult
    p = p * norm_mult
    m = m * norm_mult
    rms = rms * norm_mult

    inputs = {}

    fraction_left = fractional_tanh_embed(fraction_trained)
    inputs["fraction_left"] = fraction_left.to(device)
    inputs["loss_features"] = loss_features

    leading_axis = list(range(0, len(p.shape)))
    mean_m = torch.mean(m, dim=leading_axis, keepdim=True)
    var_m = torch.mean((m - mean_m) ** 2, dim=leading_axis)
    inputs["var_m"] = clip_log_abs(var_m, scale=10.0)

    mean_rms = torch.mean(rms, dim=leading_axis, keepdim=True)
    var_rms = torch.mean((rms - mean_m) ** 2, dim=leading_axis)
    inputs["mean_rms"] = clip_log_abs(mean_rms.view(-1), scale=10.0)
    inputs["var_rms"] = clip_log_abs(var_rms, scale=10.0)

    n_rank = sum([1 for dim in p.shape if dim > 1])
    inputs["rank"] = F.one_hot(torch.tensor(n_rank), num_classes=5).float().to(device)
    values = sorted_values(inputs)
    values = [v if len(v.shape) == 1 else v.unsqueeze(0) for v in values]
    return torch.cat(values, dim=0)


class VeLO_naive(Optimizer):
    def __init__(
        self,
        params,
        momentum_decays=[0.0, 0.0, 0.0],
        rms_decays=[0.0],
        adafactor_decays=[0.0, 0.0, 0.0],
        lr=1e-3,
        exp_mult=0.001,
        step_mult=0.001,
        input_size=30,
        hidden_size=4,
        hidden_layers=1,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        lstm_input_size=30,
        lstm_hidden_size=512,
        param_inits=256,
        num_steps=10000,
        initial_rms_decays=(0.999,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        concat_weights=True,
        make_separate_weights=False,
        split_weights=False,
        weight_decay=0.0,
        clip_grad=False,
        mup_lrs=None,
        hf_key_rnn="Pauljanson002/VeLO_RNN",
        hf_key_mlp="Pauljanson002/VeLO_MLP",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        momentum_decays = torch.tensor(momentum_decays).to(self.device)
        rms_decays = torch.tensor(rms_decays).to(self.device)
        adafactor_decays = torch.tensor(adafactor_decays).to(self.device)
        mom_decay = param_to_decay(
            decay_to_param(torch.tensor(initial_momentum_decays, device=self.device)) + momentum_decays
        )
        rms_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_rms_decays, device=self.device)) + rms_decays
        )
        adafactor_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_adafactor_decays, device=self.device)) + adafactor_decays
        )
        clip_mom_decays = torch.clip(mom_decay, 0.0, 1.0).to(self.device)
        clip_rms_decays = torch.clip(rms_decays, 0.0, 1.0).to(self.device)
        clip_adafactor_decays = torch.clip(adafactor_decays, 0.0, 1.0).to(self.device)

        defaults = dict(
            lr=lr,
            exp_mult=exp_mult,
            step_mult=step_mult,
            initial_momentum_decays=clip_mom_decays,
            lstm_hidden_size=lstm_hidden_size,
            initial_rms_decays=clip_rms_decays,
            initial_adafactor_decays=clip_adafactor_decays,
            param_inits=param_inits,
            concat_weights=concat_weights,
            make_separate_weights=make_separate_weights,
            input_size=input_size,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            split_weights=split_weights,
            clip_grad=clip_grad,
            mup_lrs=mup_lrs,
            weight_decay=weight_decay,
        )
        super(VeLO_naive, self).__init__(params, defaults)

        self.buffer_loss_fns = BufferLossAccumulators(self.device)
        self.loss_buffer = self.buffer_loss_fns.init(num_steps)
        self.num_steps = num_steps
        self.rnn = VeLORNN.from_pretrained(hf_key_rnn)
        self.lstm_init_state = self.rnn.lstm_init_state
        self.rnn.to(self.device)
        self.network_stack = VeLOMLP.from_pretrained(hf_key_mlp)
        self.network_stack.to(self.device)
        for name, param in self.network_stack.named_parameters():
            param.requires_grad = False
        for name, param in self.rnn.named_parameters():
            param.requires_grad = False
        self.init_state()

    @torch.no_grad()
    def init_state(
        self,
    ):
        layer_idx = 0
        for group in self.param_groups:
            group["step"] = 0
            for p in group["params"]:
                if p.requires_grad is False:
                    continue
                state = self.state[p]
                p_shape = p.shape
                if len(state) == 0:
                    state["layer_idx"] = layer_idx
                    state["mom"] = torch.zeros(p_shape + (3,)).to(self.device)
                    state["rms"] = torch.zeros(p_shape + (1,)).to(self.device)
                    state["fac_vec_row"], state["fac_vec_col"], state["fac_vec_v"] = (
                        init_factors(p)
                    )
                    state["fac_vec_row"], state["fac_vec_col"], state["fac_vec_v"] = (
                        state["fac_vec_row"].to(self.device),
                        state["fac_vec_col"].to(self.device),
                        state["fac_vec_v"].to(self.device),
                    )
                layer_idx += 1
        self.lstm_hidden_state = (
            self.lstm_init_state[0].repeat(layer_idx, 1).to(self.device),
            self.lstm_init_state[1].repeat(layer_idx, 1).to(self.device),
        )

    @torch.no_grad()
    def collect_rnn_outputs(self, to_lstm_from_loss):
        rnn_inputs = []
        lstm_hidden_states = []
        for group in self.param_groups:
            fraction_trained = group["step"] / self.num_steps
            for p in group["params"]:
                grad = torch.clip(p.grad, -1000.0, 1000.0)
                state = self.state[p]
                mom = state["mom"]
                rms = state["rms"]
                rnn_inputs.append(
                    lstm_features_for_tensor(
                        p,
                        grad,
                        mom,
                        rms,
                        fraction_trained,
                        to_lstm_from_loss,
                        self.device,
                    )
                )
        rnn_inputs = torch.stack(rnn_inputs)
        rnn_inputs = torch.flip(rnn_inputs, [0])
        control_params, lr_mult, self.lstm_hidden_state = self.rnn(
            rnn_inputs, self.lstm_hidden_state
        )
        return control_params, lr_mult

        # Add this method to save the loss buffer and LSTM hidden state

    def state_dict(self):
        # First get the standard optimizer state_dict
        state_dict = super(VeLO_naive, self).state_dict()

        # Add our additional state information
        state_dict["loss_buffer"] = self.loss_buffer
        state_dict["lstm_hidden_state"] = self.lstm_hidden_state
        state_dict["num_steps"] = self.num_steps

        return state_dict

    # Add this method to load the loss buffer and LSTM hidden state
    def load_state_dict(self, state_dict):
        # Extract our custom state information
        loss_buffer = state_dict.pop("loss_buffer")
        lstm_hidden_state = state_dict.pop("lstm_hidden_state")
        num_steps = state_dict.pop("num_steps")

        # Load the standard optimizer state
        super(VeLO_naive, self).load_state_dict(state_dict)

        # Restore our custom state
        self.loss_buffer = loss_buffer
        self.lstm_hidden_state = lstm_hidden_state
        self.num_steps = num_steps

    @torch.no_grad()
    def step(self, loss):
        self.loss_buffer = self.buffer_loss_fns.update(self.loss_buffer, loss)
        to_lstm_from_loss = self.buffer_loss_fns.features(self.loss_buffer)
        control_params, lr_mult = self.collect_rnn_outputs(to_lstm_from_loss)
        for group in self.param_groups:
            exp_mult = group["exp_mult"]
            step_mult = group["step_mult"]
            group["step"] += 1

            for p in group["params"]:
                beta_m = group["initial_momentum_decays"]
                beta_rms = group["initial_rms_decays"]
                beta_adafactor = group["initial_adafactor_decays"]
                weight_decay = group["weight_decay"]
                p_shape = p.shape

                if p.grad is None:
                    continue
                grad = torch.clip(p.grad, -1000.0, 1000.0)
                state = self.state[p]
                mom = state["mom"]
                rms = state["rms"]
                layer_idx = state["layer_idx"]

                batch_p = p.unsqueeze(-1)
                batch_g = grad.unsqueeze(-1)
                clipped_g = torch.clip(batch_g, -0.1, 0.1)

                axis = list(range(len(p_shape)))
                for _ in axis:
                    beta_m = beta_m[None, ...]
                    beta_rms = beta_rms[None, ...]
                    beta_adafactor = beta_adafactor[None, ...]

                mom.mul_(beta_m).add_((1 - beta_m) * batch_g)
                rms.mul_(beta_rms).add_((1 - beta_rms) * (batch_g**2))
                (
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                    fac_g,
                ) = update_factors(
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                    batch_g,
                    p_shape,
                    beta_adafactor,
                )
                fac_vec_col, fac_vec_row, fac_vec_v = (
                    state["fac_vec_col"],
                    state["fac_vec_row"],
                    state["fac_vec_v"],
                )
                rsqrt = torch.rsqrt(rms + 1e-6)
                rms_norm_g = batch_g * rsqrt
                inps = [
                    batch_g,
                    clipped_g,
                    batch_p,
                    mom,
                    rms,
                    mom * rsqrt,
                    rsqrt,
                    fac_g,
                    rms_norm_g,
                ]
                f_dims = factored_dims(p_shape)
                if f_dims is not None:
                    d1, d0 = f_dims
                    rp_row = [1] * (1 + len(p_shape))
                    rp_col = [1] * (1 + len(p_shape))
                    rp_row[d0] = p_shape[d0]
                    rp_col[d1] = p_shape[d1]
                    row_feat = fac_vec_row.unsqueeze(d0).repeat(rp_row)
                    col_feat = fac_vec_col.unsqueeze(d1).repeat(rp_col)

                    inps.extend(
                        [
                            row_feat,
                            col_feat,
                            torch.rsqrt(row_feat + 1e-8),
                            torch.rsqrt(col_feat + 1e-8),
                        ]
                    )
                    reduced_d1 = d1 - 1 if d1 > d0 else d1 #!r change
                    row_col_mean = fac_vec_row.mean(dim=reduced_d1, keepdim=True) #!r change
                    row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
                    col_factor = safe_rsqrt(fac_vec_col)
                    fac_mom_mult = (
                        mom * row_factor.unsqueeze(d0) * col_factor.unsqueeze(d1)
                    )
                    inps.append(fac_mom_mult)
                else:
                    inps.extend(
                        [
                            fac_vec_v,
                            fac_vec_v,
                            torch.rsqrt(fac_vec_v + 1e-8),
                            torch.rsqrt(fac_vec_v + 1e-8),
                        ]
                    )
                    fac_mom_mult = mom * torch.pow(fac_vec_v, -0.5)
                    inps.append(fac_mom_mult)
                inps = [second_moment_normalizer(i, axis=axis) for i in inps]
                inps = torch.cat(inps, dim=-1)
                self.network_stack.update_params(control_params[-(1 + layer_idx)])
                direction, magnitude, _ = self.network_stack(inps).split(1, dim=-1)
                # print(direction.shape, magnitude.shape, _.shape)
                # print(direction, magnitude, _)
                param_scale = torch.sqrt(torch.mean(p**2) + 1e-9)
                step = param_scale * (
                    direction * torch.exp(magnitude * exp_mult) * step_mult
                ).squeeze(-1)
                step = lr_mult[-(1 + layer_idx)] * step
                p.add_(step, alpha=-group["lr"])
                if weight_decay > 0:
                    p.add_(p, alpha=-weight_decay * group["lr"])
        return
