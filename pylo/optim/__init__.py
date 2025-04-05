from pylo.optim.AdafacLO_naive import AdafacLO_naive
from pylo.optim.MuLO_naive import MuLO_naive
from pylo.optim.Velo import VeLO

# Initialize with optimizers we know we can import
__all__ = ["AdafacLO_naive", "MuLO_naive", "VeLO"]

# Try to import CUDA-based optimizers
try:
    from pylo.optim.AdafacLO_cuda import AdafacLO_CUDA
    from pylo.optim.MuLO_cuda import MuLO_CUDA

    # Add to __all__ only if successfully imported
    __all__.extend(["AdafacLO_CUDA", "MuLO_CUDA"])
except ImportError:
    import warnings

    warnings.warn("Custom CUDA optimizers could not be imported. Using native optimizers only.")
