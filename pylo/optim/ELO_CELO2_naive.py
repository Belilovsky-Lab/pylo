"""ELO_CELO2_naive: the ELO-CELO2 learned optimizer (PyTorch).

ELO-CELO2 fuses a CELO2 MLP backbone with an ELO expert mechanism *during
meta-training*. At inference / meta-test time the expert trajectory and the IMT
losses are disabled (``meta_train=False``), so the parameter update reduces
exactly to the CELO2 forward pass: CELO2 MLP steps for 2D+ parameters and AdamW
(over the shared accumulators) for 1D parameters.

Consequently this class is a thin wrapper over :class:`CELO2_naive` that only
changes the default hyper-parameters to match the ELO-CELO2 configuration
(``config/learned_optimizer/elo_celo2.py``): nonzero weight decay and enabled
gradient clipping. The distinct learned weights live in the loaded checkpoint.
"""

from typing import Optional

from pylo.optim.CELO2_naive import CELO2_naive


class ELO_CELO2_naive(CELO2_naive):
    """Inference-time ELO-CELO2 optimizer (CELO2 forward with ELO defaults)."""

    def __init__(
        self,
        params,
        num_steps,
        # LR schedule
        init_lr=0.0,
        peak_lr=1e-3,
        warmup_steps=0,
        warmup_fraction=0.05,
        end_lr=0.0,
        # ELO-CELO2 defaults differ from CELO2 here:
        weight_decay=0.1,
        clip_grad=True,
        clip_norm=1.0,
        # AdamW for 1D params
        adam_lr_mult=1.0,
        adam_weight_decay=None,
        use_adamw_for_1d=True,
        # CELO2 backbone
        orthogonalize=True,
        ff_hidden_size=8,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult=0.0,
        rmsmult=1.0,
        param_scale_mult=False,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters=5,
        ns_eps=1e-8,
        grad_clip_val=1000.0,
        # Weights
        hf_key: Optional[str] = "DiamondXL/elo-celo2",
        checkpoint_path: Optional[str] = None,
        network=None,
    ):
        super().__init__(
            params,
            num_steps=num_steps,
            init_lr=init_lr,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            warmup_fraction=warmup_fraction,
            end_lr=end_lr,
            weight_decay=weight_decay,
            adam_lr_mult=adam_lr_mult,
            adam_weight_decay=adam_weight_decay,
            use_adamw_for_1d=use_adamw_for_1d,
            orthogonalize=orthogonalize,
            clip_grad=clip_grad,
            clip_norm=clip_norm,
            ff_hidden_size=ff_hidden_size,
            ff_hidden_layers=ff_hidden_layers,
            initial_momentum_decays=initial_momentum_decays,
            initial_rms_decays=initial_rms_decays,
            initial_adafactor_decays=initial_adafactor_decays,
            exp_mult=exp_mult,
            rmsmult=rmsmult,
            param_scale_mult=param_scale_mult,
            ns_coeffs=ns_coeffs,
            ns_iters=ns_iters,
            ns_eps=ns_eps,
            grad_clip_val=grad_clip_val,
            hf_key=hf_key,
            checkpoint_path=checkpoint_path,
            network=network,
        )
        # ELO-CELO2 evaluates the LR schedule at iteration+1 (1-indexed),
        # unlike the standalone CELO2 which is 0-indexed via the optax chain.
        self._lr_offset = 0
