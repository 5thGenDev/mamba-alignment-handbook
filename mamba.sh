#!/bin/bash

# Not recommend to source .bashrc in HPC condor
source /mnt/fast/nobackup/users/nt00601/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# For mamba - Sort out undefined-symbol issue https://github.com/state-spaces/mamba/issues/169
pip install https://github.com/state-spaces/mamba/releases/download/causal_conv1d-1.1.1+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x64_64.whl
pip install https://github.com/state-spaces/mamba/releases/download/mamba_ssm-1.1.1+cu118torch2.1cxx11abiFALSE-cp38-cp38-linux_x86_64.whl

