from setuptools import setup, find_packages

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
    python_requires='>=3.6',
)