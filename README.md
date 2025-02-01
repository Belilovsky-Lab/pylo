# Installation Guide

## Prerequisites
- Set CUDA_HOME environment variable (if using CUDA)

## Installation Options

### Basic Installation (without CUDA)
```bash
pip install .
```

### CUDA-enabled Installation (This may take a while)
```bash
pip install . --config-settings="--build-option=--cuda"
```