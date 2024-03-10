#!/bin/bash

# Not recommend to source .bashrc in HPC condor
source /mnt/fast/nobackup/users/nt00601/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# For mamba
pip install causal-conv1d>=1.2.0
pip install mamba-ssm

