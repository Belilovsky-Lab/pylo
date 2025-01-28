from setuptools import setup, find_packages
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
    ],
}

setup(
    name="pylo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch==2.4.1",
        "torchvision==0.19.1",
        "torchaudio==2.4.1",
        "huggingface_hub",
        "safetensors==0.4.5",
        "mup==1.0.0",
    ],
    author="Paul Janson",
    description="A package for Pylo project",
    license="Apache-2.0",
    # url="https://github.com/Pauljanson002/test",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    ext_modules=[
        CUDAExtension(
            name="cuda_lo",
            sources=["pylo/csrc/learned_optimizer.cu"],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
