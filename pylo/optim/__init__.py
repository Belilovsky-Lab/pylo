from pylo.optim.AdafacLO_naive import AdafacLO_naive
from pylo.optim.MuLO_naive import MuLO_naive
from pylo.optim.Velo_naive import VeLO_naive
from pylo.optim.CELO2_naive import CELO2_naive
from pylo.optim.ELO_CELO2_naive import ELO_CELO2_naive
from pylo.optim.ELO_naive import ELO_naive

# Initialize with optimizers we know we can import
__all__ = [
    "AdafacLO_naive",
    "MuLO_naive",
    "VeLO_naive",
    "CELO2_naive",
    "ELO_CELO2_naive",
    "ELO_naive",
]

# CELO2 / ELO-CELO2 default to the naive (pure-PyTorch) implementation; the
# dedicated try-block below overrides these to the CUDA variants when the
# celo2_cuda_kernel extension is available (mirroring VeLO / AdafacLO).
CELO2 = CELO2_naive
ELO_CELO2 = ELO_CELO2_naive
ELO = ELO_naive
__all__.extend(["CELO2", "ELO_CELO2", "ELO"])

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

# CELO2 / ELO-CELO2 CUDA are wired independently: they only need the
# celo2_cuda_kernel extension, so a missing/failed build here leaves the other
# CUDA optimizers (and the naive CELO2 fallback set above) untouched.
try:
    from pylo.optim.CELO2_cuda import CELO2_CUDA
    from pylo.optim.ELO_CELO2_cuda import ELO_CELO2_CUDA

    CELO2 = CELO2_CUDA
    ELO_CELO2 = ELO_CELO2_CUDA
    __all__.extend(["CELO2_CUDA", "ELO_CELO2_CUDA"])
except ImportError:
    pass  # keep the naive CELO2 / ELO_CELO2 aliases set above

# ELO CUDA reuses the cuda_lo kernel (shared with AdafacLO); wire it independently
# so a failed AdafacLO_cuda import (its optional deps) doesn't disable ELO_CUDA.
try:
    from pylo.optim.ELO_cuda import ELO_CUDA

    ELO = ELO_CUDA
    __all__.append("ELO_CUDA")
except ImportError:
    pass  # keep the naive ELO alias set above
