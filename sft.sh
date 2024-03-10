#!/bin/bash

# Not recommend to source .bashrc in HPC condor
source /mnt/fast/nobackup/users/nt00601/miniconda3/etc/profile.d/conda.sh
conda activate handbook

# standard template: ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes={num_gpus} scripts/run_{task}.py recipes/{model_name}/{task}/config_qlora.yaml --load_in_4bit=false
# recipes/{model_name}/{task}/config.yaml is just directory to the yaml file itself.

# global batch size = 32, see https://github.com/huggingface/alignment-handbook/issues/45#issuecomment-1845598205
# Step 1 - SFT for mamba
# mamba uses float16 to float32 to need to look into that
# gradient_accumulation_steps=2 is default value but might need to change
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=8 scripts/run_sft mamba-sft-lora.yaml
