#!/bin/bash

# Not recommend to source .bashrc in HPC condor
source /mnt/fast/nobackup/users/nt00601/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# For mamba - Sort out undefined-symbol issue https://github.com/state-spaces/mamba/issues/169
pip install causal_conv1d-1.0.1+cu188torch2.1cxx11abiFALSE-cp310-cp310-linux_x64_64.whl
pip install mamba_ssm-1.0.1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

