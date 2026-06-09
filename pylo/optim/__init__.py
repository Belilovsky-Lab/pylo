from pylo.optim.AdafacLO_naive import AdafacLO_naive
from pylo.optim.MuLO_naive import MuLO_naive
from pylo.optim.Velo_naive import VeLO_naive
from pylo.optim.CELO2_naive import CELO2_naive
from pylo.optim.ELO_CELO2_naive import ELO_CELO2_naive

# Initialize with optimizers we know we can import
__all__ = [
    "AdafacLO_naive",
    "MuLO_naive",
    "VeLO_naive",
    "CELO2_naive",
    "ELO_CELO2_naive",
]

# CELO2 / ELO-CELO2 currently ship a naive (pure-PyTorch) implementation only.
# Expose the bare public aliases now; if a CUDA build is added later these can be
# overridden in the try-block below, mirroring VeLO / AdafacLO.
CELO2 = CELO2_naive
ELO_CELO2 = ELO_CELO2_naive
__all__.extend(["CELO2", "ELO_CELO2"])

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
