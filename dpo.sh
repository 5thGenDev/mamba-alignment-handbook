#!/bin/bash

# Not recommend to source .bashrc in HPC condor
source /mnt/fast/nobackup/users/nt00601/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# standard template: ACCELERATE_LOG_LEVEL=info accelerate launch --config_file path/to/multi_gpu.yaml --num_processes={num_gpus} path/to/scripts/run_sft.py path/to/config_qlora.yaml
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /mnt/fast/nobackup/users/nt00601/mamba-alignment-handbook/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 /mnt/fast/nobackup/users/nt00601/mamba-alignment-handbook/scripts/run_dpo.py /mnt/fast/nobackup/users/nt00601/mamba-alignment-handbook/mamba-dpo-qlora.yaml
