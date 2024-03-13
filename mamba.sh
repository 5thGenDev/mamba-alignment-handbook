#!/bin/bash

# Not recommend to source .bashrc in HPC condor
source /mnt/fast/nobackup/users/nt00601/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# For mamba - Sort out undefined-symbol issue https://github.com/state-spaces/mamba/issues/169
# mamba only supports CUDA 11.8 and CUDA 12.2, so pick according to what your condor have
# cp310 = python version 3.10
# torch 2.1 = Pytorch 2.10
# Have a look at this and google which tag is compatible with CUDA 11.8 or CUDA 12.2
pip install https://github.com/state-spaces/mamba/releases/download/causal_conv1d-1.1.1+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x64_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/mamba_ssm-1.1.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

