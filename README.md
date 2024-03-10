Please see instruction for original alignment-handbook here: https://github.com/huggingface/alignment-handbook/tree/main

### Overall pipeline: 
1. Install necessary packages and pull docker image to host pretrained state-spaces/mamba-2.8b-slimpj.
2. Finetune pretrained mamba on SFT training script according to alignment-handbook github. 
3. Further finetune it on DPO training script according to alignment-handbook github.
4. Evaluate on MTBench and AlpacaEval.
To access A100 GPUs, all finetunings were done HPC Condor so I was in charge for (1) and writing Slurm scripts in .submit_file for (2) and (3). Meanwhile everyone including me investigated across many Git issue posts of Mamba and Zephyr in order to nail down the exact parameters to be tweaked when adapting Mamba onto SFT, DPO finetuning scripts.

### Reproducible pipeline: Expect a lot of bug at the moment.
1. Git clone based alignment-handbook and based mamba and follow their installation instructions to get all necessary dependencies
2. Delete git clone of based alignment-handbook and based mamba
3. Git clone my modified mamba and alignment-handbook: https://github.com/5thGenDev/mamba-finetune/tree/main and https://github.com/5thGenDev/mamba-alignment-handbook
4. Read the instruction in https://github.com/5thGenDev/mamba-finetune/blob/main/README.md carefully because it tells you which .py to make manual adjustment yourself.
