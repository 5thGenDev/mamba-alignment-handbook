Please see instruction for original alignment-handbook here: https://github.com/huggingface/alignment-handbook/tree/main

### Overall pipeline: 
1. Install necessary packages and pull docker image to host pretrained state-spaces/mamba-2.8b-slimpj.
2. Finetune pretrained mamba on SFT training script according to alignment-handbook github. 
3. Further finetune it on DPO training script according to alignment-handbook github.
4. Evaluate on MTBench and AlpacaEval.
In order to get access to A100 GPUs, all finetuning happen on HPC Condor, which mean I was in charge for (1) and writing Slurm scripts in .submit_file for (2) and (3). Meanwhile everyone including me investigated across many Git issue posts of Mamba and Zephyr in order to nail down the exact parameters to be tweaked when adapting Mamba onto SFT, DPO finetuning scripts.
