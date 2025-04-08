"""MuLO_CUDA: An Cuda-accelerated MLP learned optimizer in μP.

This is a PyTorch implementation of μLO from: https://arxiv.org/abs/2406.00153

The following code is adapted from the following Jax implementation: https://github.com/bentherien/mu_learned_optimization/blob/main/src/mup_adafac_mlp_lopt.py
"""
from mup.optim import process_param_groups
from collections import defaultdict
from pylo.optim import AdafacLO_naive


def MuLO_naive(params, impl=AdafacLO_naive, **kwargs):
    """
    μP (Maximal Update Parameterization) wrapper for the PyTorch native implementation
    of the Adafac learned optimizer.

    This function applies the μP parameterization to the Adafac learned optimizer,
    scaling learning rates for matrix-like parameters according to their width multipliers.
    Parameters are organized into groups based on their infinite-width shape properties.

    Note:
        This implementation requires that all parameters have been processed with
        mup.set_base_shapes() to establish their infinite-width behavior.

    Example:
        >>> model = MyModel()
        >>> mup.set_base_shapes(model, base_model)
        >>> optimizer = MuLO_naive(model.parameters())
    """
    new_param_groups = []
    for param_group in process_param_groups(params, **kwargs):
        # For every existing param group, we split into several new groups
        def new_group():
            new_g = {k: v for k, v in param_group.items() if k != "params"}
            new_g["params"] = []
            return new_g

        # The matrix-like weights might need multiple groups since weights
        # might have different width multipliers
        matrix_like_p = defaultdict(new_group)  # key is width_mult
        vector_like_p = new_group()
        for p in param_group["params"]:
            # print(p.infshape.width_mult())
            assert hasattr(p, "infshape"), (
                f"A parameter with shape {p.shape} does not have `infshape` attribute. "
                "Did you forget to call `mup.set_base_shapes` on the model?"
            )
            if p.infshape.ninf() == 2:
                matrix_like_p[p.infshape.width_mult()]["params"].append(p)
            elif p.infshape.ninf() > 2:
                raise NotImplementedError("more than 2 inf dimensions")
            else:
                vector_like_p["params"].append(p)
        for width_mult, group in matrix_like_p.items():
            # Scale learning rate and weight decay accordingly
            group["lr"] /= width_mult
        new_param_groups.extend(list(matrix_like_p.values()) + [vector_like_p])
    return impl(new_param_groups, **kwargs)
