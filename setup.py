from setuptools import setup, find_packages
import sys


def get_build_config():
    # Parse config settings from --config-settings
    # pip passes config settings as --config-settings="--build-option=--cuda"
    cuda_setting = False
    for arg in sys.argv:
        if arg.startswith('--build-option=--cuda'):
            cuda_setting = True
            sys.argv.remove(arg)
            break
        elif arg == '--cuda':  # For direct python setup.py install
            cuda_setting = True
            sys.argv.remove(arg)
            break
    return cuda_setting

# Check build configuration
enable_cuda = get_build_config()
if enable_cuda:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension



# Advanced optimization flags
extra_compile_args = {
    "cxx": [
        "-O3",  # Maximum optimization
        "-DNDEBUG",  # Disable debug assertions
        "-march=native",  # Optimize for host CPU
        "-ffast-math",  # Fast math operations
        "-funroll-loops",  # Loop unrolling
        "-flto",  # Link-time optimization
        "-fomit-frame-pointer",  # Remove frame pointers
        # "-std=c++17"
    ],
    "nvcc": [
        "-O3",  # Maximum CUDA optimization
        "--use_fast_math",  # Fast math for CUDA
        "-DNDEBUG",  # Disable debug assertions
        # "--gpu-architecture=sm_89",  # Target recent GPU arch
        "--ftz=true",  # Flush denormals to zero
        "--prec-div=false",  # Fast division
        "--prec-sqrt=false",  # Fast square root
        # "--maxrregcount=128",    # Limit registers for better occupancy
        # "-Wl,-rpath,/usr/lib/x86_64-linux-gnu"
    ],
}

# Prepare extension modules based on cuda flag
ext_modules = []
cmdclass = {}

if enable_cuda:
    ext_modules.append(
        CUDAExtension(
            name="cuda_lo",
            sources=["pylo/csrc/learned_optimizer.cu"],
            extra_compile_args=extra_compile_args,
        )
    )
    ext_modules.append(
        CUDAExtension(
            name="velo_cuda_kernel",
            sources=["pylo/csrc/velo_kernel.cu"],
            extra_compile_args=extra_compile_args,
        )
    )
    cmdclass["build_ext"] = BuildExtension

setup(
    name="pylo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "torchaudio",
        "huggingface_hub",
        "safetensors==0.4.5",
        "mup==1.0.0",
    ],
    author="Paul Janson",
    description="A package for Pylo project",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)


