from pylo.optim.AdafacLO_naive import AdafacLO_naive
from pylo.optim.MuLO_naive import MuLO_naive
from pylo.optim.Velo_naive import VeLO_naive
from pylo.optim.velo_cuda import VeLO_CUDA

# Initialize with optimizers we know we can import
__all__ = ["AdafacLO_naive", "MuLO_naive", "VeLO_naive"]

# Try to import CUDA-based optimizers
try:
    from pylo.optim.AdafacLO_cuda import AdafacLO_CUDA
    from pylo.optim.MuLO_cuda import MuLO_CUDA
    from pylo.optim.velo_cuda import VeLO_CUDA

    # Add to __all__ only if successfully imported
    __all__.extend(["AdafacLO_CUDA", "MuLO_CUDA", "VeLO_CUDA"])
except ImportError:
    import warnings

    warnings.warn("Custom CUDA optimizers could not be imported. Using native optimizers only.")
