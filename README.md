Please see instruction for original alignment-handbook here: https://github.com/huggingface/alignment-handbook/tree/main
***We aren't the first ppl trying to finetune mamba. Other had tried before: https://github.com/state-spaces/mamba/pull/83, https://github.com/havenhq/mamba-chat, https://github.com/geronimi73/mamba/blob/main/finetune.py, Incorporate all lessons learnt all existing githubs and extensively reading model cards, Huggingface script and research papers from Mamba and alignment-handbook, we genuinely believe that we have produced a better finetuned mamba LLM.

### Overall pipeline: 
1. Install necessary packages and pull docker image to host pretrained state-spaces/mamba-2.8b-slimpj.
2. Finetune pretrained mamba on SFT training script according to alignment-handbook github. 
3. Further finetune it on DPO training script according to alignment-handbook github. Specifically these twos: https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta, https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-gemma
4. Evaluate on MTBench and AlpacaEval according to the instruction from alignment-handbook research paper: https://arxiv.org/pdf/2310.16944.pdf section 4.2
To access A100 GPUs, all finetunings were done HPC Condor so I was in charge for (1) and writing Slurm scripts in .submit_file for (2) and (3). Meanwhile everyone including me investigated across many Git issue posts of Mamba and Zephyr in order to nail down the exact parameters to be tweaked when adapting Mamba onto SFT, DPO finetuning scripts.

### Why did we fail
Our HPC condor frequently has server issues where:
- Slurm handles things by different queues but if you're not careful, it can leave resources sitting empty 'just in case'. This happened to us in this project.
- Multi-GPU jobs only gets extra priority when a machine (e.g. A100 compute node) has just come back up from draining, for the first few minutes they 'prefer' multi GPU jobs. This is by design because we found that machines were getting full of single GPU jobs and the multi jobs couldn't get slots. Now we run a 'defrag' system that picks a machine to drain every x hours so that multi gpu jobs can run. Likely happened to us too.
- Some people have been potentially abusing the condor job scheduler to reserve compute resources for themselves by bypassing the limitations set on interactive jobs by submitting regular batch jobs with processes that idle but keep the job running so they can use the condor ssh to job feature to connect to their job and use it interactively. Maybe, but not likely.

### Reproducible pipeline: Expect a lot of bugS at the moment because we couldn't test it on our HPC condor
1. Git clone based alignment-handbook and based mamba and follow their installation instructions to get all necessary dependencies
2. Delete git clone of based alignment-handbook and based mamba
3. Git clone my modified mamba and alignment-handbook: https://github.com/5thGenDev/mamba-finetune/tree/main and https://github.com/5thGenDev/mamba-alignment-handbook. Based mamba can't finetune on downstream task and based-alignment-handbook has different config than mamba config even though they both inherit from HuggingFace, so some modified code is needed.
4. Read the instruction in https://github.com/5thGenDev/mamba-finetune/blob/main/README.md carefully because it tells you which .py to make manual adjustment yourself.

