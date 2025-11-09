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
import velo_cuda_kernel
import time

from pylo.models.VeLO_MLP import VeLOMLP
from pylo.models.VeLO_RNN import VeLORNN


def init_factors(p):
    shape = p.shape
    f_dims = factored_dims(shape)
    # Place multiple momentum dimension first: (3,) + shape
    shape_with_batch = (3,) + shape
    if f_dims is not None:
        dc, dr = f_dims
        # Adjust indices since we now have a leading dimension
        # d0 and d1 need to be offset by 1
        vr_shape = list(shape)
        vr_shape[dr] = 1
        vr_shape = [3] + vr_shape
        vc_shape = list(shape)
        vc_shape[dc] = 1
        vc_shape = [3] + vc_shape
        v_row = torch.zeros(vr_shape, dtype=torch.float32)
        v_col = torch.zeros(vc_shape, dtype=torch.float32)
        return v_row, v_col, torch.tensor([], dtype=torch.float32)

    else:
        v = torch.zeros(shape_with_batch, dtype=torch.float32)
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
    # g has shape [...] (original grad shape)
    grad_sqr = g * g + epsilon

    if f_dims is not None:
        d1, d0 = f_dims
        # Reshape decay_rate for broadcasting: [n_decays, 1, 1, ...]
        decay_view = decay_rate.view(-1, *[1] * len(g_shape))
        mixing_view = mixing_rate.view(-1, *[1] * len(g_shape))

        # Mean over grad dimensions, then add leading dimension for broadcasting
        new_v_row = (
            decay_view * v_row
            + mixing_view * grad_sqr.mean(dim=d0, keepdim=True)[None, ...]
        )
        new_v_col = (
            decay_view * v_col
            + mixing_view * grad_sqr.mean(dim=d1, keepdim=True)[None, ...]
        )

        # reduced_d1 needs +1 offset because of leading dimension in v_row
        reduced_d1 = (d1 + 1) - 1 if (d1 + 1) > (d0 + 1) else (d1 + 1)
        row_col_mean = torch.mean(new_v_row, dim=reduced_d1, keepdim=True)

        row_factor = safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
        col_factor = safe_rsqrt(new_v_col)
        # Broadcast g to [n_decays, ...] then multiply with factors
        y = g[None, ...] * row_factor.unsqueeze(d0 + 1) * col_factor.unsqueeze(d1 + 1)
        return new_v_col, new_v_row, torch.tensor([], dtype=torch.float32), y

    else:
        decay_view = decay_rate.view(-1, *[1] * len(g_shape))
        mixing_view = mixing_rate.view(-1, *[1] * len(g_shape))

        new_v = decay_view * v_full + mixing_view * grad_sqr[None, ...]
        y = g[None, ...] * safe_rsqrt(new_v + 1e-9)
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

@torch.compile
def lstm_features_for_tensor(p, g, m, rms, fraction_left, loss_features, rank_onehot, device):
    # Timing: Normalization
    norm_mult = torch.rsqrt(torch.clamp(torch.mean(p**2), min=1e-9))
    g = g * norm_mult
    p = p * norm_mult
    m = m * norm_mult
    rms = rms * norm_mult

    # Pre-allocate result tensor with total size 30
    # Feature layout: ['fraction_left' 0:9, 'loss_features' 9:18, 'mean_rms' 18:19, 'rank' 19:24, 'var_m' 24:27, 'var_rms' 27:30]
    result = torch.empty(30, device=device, dtype=torch.float32)

    # Timing: Fraction features (now just copying pre-computed values)

    result[0:9] = fraction_left
    result[9:18] = loss_features


    # Timing: Momentum features

    leading_axis = list(range(1, len(p.shape)+1))
    mean_m = torch.mean(m, dim=leading_axis, keepdim=True)
    var_m = torch.mean((m - mean_m) ** 2, dim=leading_axis)
    result[24:27] = clip_log_abs(var_m, scale=10.0)


    # Timing: RMS features

    mean_rms = torch.mean(rms, dim=leading_axis, keepdim=True)
    var_rms = torch.mean((rms - mean_m) ** 2, dim=leading_axis)
    result[18:19] = clip_log_abs(mean_rms.view(-1), scale=10.0)
    result[27:30] = clip_log_abs(var_rms, scale=10.0)


    # Use pre-computed rank one-hot encoding
    result[19:24] = rank_onehot


    return result


class VeLO_CUDA(Optimizer):
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
        legacy=False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        momentum_decays = torch.tensor(momentum_decays).to(self.device)
        rms_decays = torch.tensor(rms_decays).to(self.device)
        adafactor_decays = torch.tensor(adafactor_decays).to(self.device)
        mom_decay = param_to_decay(
            decay_to_param(torch.tensor(initial_momentum_decays, device=self.device))
            + momentum_decays
        )
        rms_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_rms_decays, device=self.device))
            + rms_decays
        )
        adafactor_decays = param_to_decay(
            decay_to_param(torch.tensor(initial_adafactor_decays, device=self.device))
            + adafactor_decays
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
            legacy=legacy,
        )
        super(VeLO_CUDA, self).__init__(params, defaults)

        self.legacy = legacy
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

        # Initialize second moment buffer for CUDA kernel
        if not self.legacy:
            self.second_moment = torch.zeros(
                30, dtype=torch.float32, device=self.device
            )

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
                    state["mom"] = torch.zeros((3,) + p_shape).to(self.device)
                    state["rms"] = torch.zeros((1,) + p_shape).to(self.device)
                    state["fac_vec_row"], state["fac_vec_col"], state["fac_vec_v"] = (
                        init_factors(p)
                    )
                    state["fac_vec_row"], state["fac_vec_col"], state["fac_vec_v"] = (
                        state["fac_vec_row"].to(self.device),
                        state["fac_vec_col"].to(self.device),
                        state["fac_vec_v"].to(self.device),
                    )
                    # Pre-compute rank one-hot encoding (static for each parameter)
                    n_rank = sum([1 for dim in p_shape if dim > 1])
                    state["rank_onehot"] = F.one_hot(torch.tensor(n_rank), num_classes=5).float().to(self.device)
                layer_idx += 1
        self.lstm_hidden_state = (
            self.lstm_init_state[0].repeat(layer_idx, 1).to(self.device),
            self.lstm_init_state[1].repeat(layer_idx, 1).to(self.device),
        )

    @torch.no_grad()
    def collect_rnn_outputs(self, to_lstm_from_loss):
        rnn_inputs = []
        lstm_hidden_states = []

        # Timing: Feature extraction loop
        start_feature_loop = time.perf_counter()
        for group in self.param_groups:
            fraction_trained = group["step"] / self.num_steps
            # Pre-compute fraction_left once per group (same for all parameters in group)
            fraction_left = fractional_tanh_embed(fraction_trained).to(self.device)
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
                        fraction_left,
                        to_lstm_from_loss,
                        state["rank_onehot"],
                        self.device,
                    )
                )
        feature_loop_time = time.perf_counter() - start_feature_loop

        # Timing: Stack and flip
        start_stack = time.perf_counter()
        rnn_inputs = torch.stack(rnn_inputs)
        rnn_inputs = torch.flip(rnn_inputs, [0])
        stack_time = time.perf_counter() - start_stack

        # Timing: RNN forward pass
        start_rnn_forward = time.perf_counter()
        control_params, lr_mult, self.lstm_hidden_state = self.rnn(
            rnn_inputs, self.lstm_hidden_state
        )
        rnn_forward_time = time.perf_counter() - start_rnn_forward

        #print(f"  RNN breakdown - Features: {feature_loop_time*1000:.3f}ms | Stack: {stack_time*1000:.3f}ms | Forward: {rnn_forward_time*1000:.3f}ms")

        return control_params, lr_mult

        # Add this method to save the loss buffer and LSTM hidden state

    def state_dict(self):
        # First get the standard optimizer state_dict
        state_dict = super(VeLO_CUDA, self).state_dict()

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
        super(VeLO_CUDA, self).load_state_dict(state_dict)

        # Restore our custom state
        self.loss_buffer = loss_buffer
        self.lstm_hidden_state = lstm_hidden_state
        self.num_steps = num_steps

    @torch.no_grad()
    def step(self, loss):
        # Timing: Buffer update
        start_buffer = time.perf_counter()
        self.loss_buffer = self.buffer_loss_fns.update(self.loss_buffer, loss)
        to_lstm_from_loss = self.buffer_loss_fns.features(self.loss_buffer)
        buffer_time = time.perf_counter() - start_buffer

        # Timing: RNN forward pass
        start_rnn = time.perf_counter()
        control_params, lr_mult = self.collect_rnn_outputs(to_lstm_from_loss)
        rnn_time = time.perf_counter() - start_rnn

        # Timing: Kernel/step loop
        start_kernel = time.perf_counter()
        if self.legacy:
            result = self._step_loop(control_params, lr_mult, self._process_param_legacy)
        else:
            result = self._step_loop(control_params, lr_mult, self._process_param_kernel)
        kernel_time = time.perf_counter() - start_kernel

        # Print timings
        #print(f"Buffer: {buffer_time*1000:.3f}ms | RNN: {rnn_time*1000:.3f}ms | Kernel: {kernel_time*1000:.3f}ms")

        return result

    def _step_loop(self, control_params, lr_mult, param_processor):
        """Common parameter loop for both implementations."""
        for group in self.param_groups:
            exp_mult = group["exp_mult"]
            step_mult = group["step_mult"]
            weight_decay = group["weight_decay"]
            group["step"] += 1

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = torch.clip(p.grad, -1000.0, 1000.0)
                state = self.state[p]
                mom = state["mom"]
                rms = state["rms"]
                layer_idx = state["layer_idx"]
                p_shape = p.shape

                # Get decay parameters
                beta_m = group["initial_momentum_decays"]
                beta_rms = group["initial_rms_decays"]
                beta_adafactor = group["initial_adafactor_decays"]

                # Prepare batched gradient (now dimension is first)
                batch_g = grad[None, ...]

                # Expand decay parameters for broadcasting
                axis = list(range(len(p_shape)))
                # Reshape betas to (n_decays, 1, 1, ...) for broadcasting
                beta_m_view = beta_m.view(-1, *[1] * len(p_shape))
                beta_rms_view = beta_rms.view(-1, *[1] * len(p_shape))
                beta_adafactor_view = beta_adafactor.view(-1, *[1] * len(p_shape))

                # Update momentum (broadcasting over first dimension)
                mom.lerp_(batch_g, 1 - beta_m_view.to(grad.dtype))

                # Update RMS (broadcasting over first dimension)
                rms.lerp_(batch_g**2, 1 - beta_rms_view.to(grad.dtype))

                f_dims = factored_dims(p_shape)
                grad_sqr = torch.square(grad) + 1e-30
                if f_dims is not None:
                    dc, dr = f_dims
                    state["fac_vec_row"].lerp_(
                        grad_sqr.mean(dim=dr, keepdim=True)[None, ...],
                        1 - beta_adafactor_view.to(state["fac_vec_row"].dtype),
                    )
                    state["fac_vec_col"].lerp_(
                        grad_sqr.mean(dim=dc, keepdim=True)[None, ...],
                        1 - beta_adafactor_view.to(state["fac_vec_col"].dtype),
                    )
                else:
                    state["fac_vec_v"].lerp_(
                        grad_sqr[None, ...],
                        1 - beta_adafactor_view.to(state["fac_vec_v"].dtype),
                    )

                # Call the parameter processor function
                param_processor(
                    p=p,
                    grad=grad,
                    state=state,
                    mom=mom,
                    rms=rms,
                    p_shape=p_shape,
                    axis=axis,
                    layer_idx=layer_idx,
                    control_params=control_params,
                    lr_mult=lr_mult,
                    group=group,
                    exp_mult=exp_mult,
                    step_mult=step_mult,
                    weight_decay=weight_decay,
                )
        return

    def _process_param_legacy(
        self,
        p,
        grad,
        batch_g,
        state,
        mom,
        rms,
        fac_g,
        p_shape,
        axis,
        layer_idx,
        control_params,
        lr_mult,
        group,
        exp_mult,
        step_mult,
        weight_decay,
    ):
        """Original Python-based parameter processing."""
        batch_p = p[None, ...]  # Now batch dimension is first
        clipped_g = torch.clip(batch_g, -0.1, 0.1)

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
            # Adjust for leading batch dimension
            d0_adj = d0 + 1
            d1_adj = d1 + 1
            rp_row = [1] * (1 + len(p_shape))
            rp_col = [1] * (1 + len(p_shape))
            rp_row[d0_adj] = p_shape[d0]
            rp_col[d1_adj] = p_shape[d1]
            row_feat = fac_vec_row.unsqueeze(d0_adj).repeat(rp_row)
            col_feat = fac_vec_col.unsqueeze(d1_adj).repeat(rp_col)

            inps.extend(
                [
                    row_feat,
                    col_feat,
                    torch.rsqrt(row_feat + 1e-8),
                    torch.rsqrt(col_feat + 1e-8),
                ]
            )
            reduced_d1 = (d1 + 1) - 1 if (d1 + 1) > (d0 + 1) else (d1 + 1)
            row_col_mean = fac_vec_row.mean(dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(fac_vec_col)
            fac_mom_mult = (
                mom * row_factor.unsqueeze(d0_adj) * col_factor.unsqueeze(d1_adj)
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
        # Adjust axis for second moment normalization (skip batch dimension)
        axis_for_norm = [i + 1 for i in axis]
        second_moment_sum = [
            torch.sum(torch.square(i), dim=axis_for_norm) for i in inps
        ]
        inps = [second_moment_normalizer(i, axis=axis_for_norm) for i in inps]
        inps = torch.cat(inps, dim=0)  # Cat along batch dimension
        self.network_stack.update_params(control_params[-(1 + layer_idx)])
        direction, magnitude, _ = self.network_stack(inps).split(1, dim=0)
        param_scale = torch.sqrt(torch.mean(p**2) + 1e-9)
        step = param_scale * (
            direction * torch.exp(magnitude * exp_mult) * step_mult
        ).squeeze(0)
        step = lr_mult[-(1 + layer_idx)] * step
        if True:
            p.add_(step, alpha=-group["lr"])
            if weight_decay > 0:
                p.add_(p, alpha=-weight_decay * group["lr"])
        return direction * torch.exp(magnitude * exp_mult) * step_mult

    def _process_param_kernel(
        self,
        p,
        grad,
        state,
        mom,
        rms,
        p_shape,
        axis,
        layer_idx,
        control_params,
        lr_mult,
        group,
        exp_mult,
        step_mult,
        weight_decay,
    ):
        """CUDA kernel-based parameter processing."""
        self.second_moment.zero_()

        f_dims = factored_dims(p_shape)

        # Get MLP weights from control_params
        self.network_stack.update_params(control_params[-(1 + layer_idx)])
        mlp_params = dict(self.network_stack.named_parameters())
        input_weights = mlp_params["input_weights"].data
        input_bias = mlp_params["input_bias"].data
        hidden_weights = mlp_params["hidden_weights.0"].data
        hidden_bias = mlp_params["hidden_bias.0"].data
        output_weights = mlp_params["output_weights"].data
        output_bias = mlp_params["output_bias"].data

        # Calculate row_factor and col_factor from factored accumulators
        if f_dims is not None:
            d1, d0 = f_dims
            dc, dr = d1, d0  # column dimension, row dimension
            vector_like = 0  # factored mode
            fac_vec_row = state["fac_vec_row"]
            fac_vec_col = state["fac_vec_col"]

            # Calculate row_factor: safe_rsqrt(fac_vec_row / row_col_mean)
            # adjusted indices for leading dimension
            reduced_d1 = (d1 + 1) - 1 if (d1 + 1) > (d0 + 1) else (d1 + 1)
            row_col_mean = torch.mean(fac_vec_row, dim=reduced_d1, keepdim=True)
            row_factor = safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
            col_factor = safe_rsqrt(fac_vec_col)

            # No permute needed - dimensions are already in the right order
        else:
            # Non-factored case
            dc, dr = 0, 0  # no factored dimensions
            vector_like = 1  # vector mode
            fac_vec_row = state["fac_vec_v"]
            fac_vec_col = state["fac_vec_v"]
            row_factor = safe_rsqrt(fac_vec_row + 1e-9)
            col_factor = torch.ones_like(row_factor)

        velo_cuda_kernel.velo_kernel_simple(
            grad,  # gradient
            p,  # parameters
            mom,  # momentum
            rms,  # RMS
            row_factor,  # row scaling factors
            col_factor,  # column scaling factors
            fac_vec_row,  # factored row accumulator
            fac_vec_col,  # factored column accumulator
            self.second_moment,  # second moment buffer (size 30)
            input_weights,  # MLP input weights
            input_bias,  # MLP input bias
            hidden_weights,  # MLP hidden weights
            hidden_bias,  # MLP hidden bias
            output_weights,  # MLP output weights
            output_bias,  # MLP output bias
            group["lr"] * lr_mult[-(1 + layer_idx)].item(),  # learning rate
            step_mult,  # step multiplier
            exp_mult,  # exp multiplier
            1e-6,  # epsilon
            weight_decay,  # weight decay
            dc,  # column dimension (0 if not factored)
            dr,  # row dimension (0 if not factored)
            vector_like,  # 0 for factored, 1 for vector mode
        )
