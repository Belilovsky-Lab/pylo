from pylo.optim.AdafacLO_naive import AdafacLO_naive
from pylo.optim.MuLO_naive import MuLO_naive
from pylo.optim.Velo_naive import VeLO_naive

# Initialize with optimizers we know we can import
__all__ = ["AdafacLO_naive", "MuLO_naive", "VeLO_naive"]

# Try to import CUDA-based optimizers
try:
    from pylo.optim.AdafacLO_cuda import AdafacLO_CUDA
    from pylo.optim.MuLO_cuda import MuLO_CUDA
    from pylo.optim.velo_cuda import VeLO_CUDA

    # Public aliases: by default, these names refer to the CUDA
    # implementations (which is what example scripts and the unit tests
    # import). If the CUDA extensions fail to load we fall back to the
    # naive implementations below so the aliases remain usable.
    VeLO = VeLO_CUDA
    AdafacLO = AdafacLO_CUDA
    MuLO = MuLO_CUDA

    # Add to __all__ only if successfully imported
    __all__.extend([
        "AdafacLO_CUDA", "MuLO_CUDA", "VeLO_CUDA",
        "VeLO", "AdafacLO", "MuLO",
    ])
except ImportError:
    import warnings

    warnings.warn("Custom CUDA optimizers could not be imported. Using native optimizers only.")

    VeLO = VeLO_naive
    AdafacLO = AdafacLO_naive
    MuLO = MuLO_naive
    __all__.extend(["VeLO", "AdafacLO", "MuLO"])
