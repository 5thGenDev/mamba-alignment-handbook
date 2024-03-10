Please see instruction for original alignment-handbook here: https://github.com/huggingface/alignment-handbook/tree/main


### Overall pipeline: 
1. Install necessary packages and pull docker image to host pretrained state-spaces/mamba-2.8b-slimpj.
2. Finetune pretrained mamba on SFT and DPO training script according to alignment-handbook github. Specifically these twos: https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta, https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-gemma
4. Evaluate on MTBench and AlpacaEval according to the instruction from alignment-handbook research paper: https://arxiv.org/pdf/2310.16944.pdf section 4.2
***We are half stucked on (2) (3) because condor decided not to run my Slurm script (see [Why did we fail section](https://github.com/5thGenDev/mamba-alignment-handbook/blob/main/README.md#why-did-we-fail) below)***. To access A100 GPUs, all finetunings were done HPC Condor so I was in charge for (1) and writing Slurm scripts in .submit_file for (2) and (3). Meanwhile everyone including me investigated across many Git issue posts of Mamba and Zephyr in order to nail down the exact parameters to be tweaked when adapting Mamba onto SFT, DPO finetuning scripts.


### But why mamba?
1. ***If all human can read 1500 words in seconds and have even shorter memories than that movie "50 first dates", then LLM can be considered as humans because [all LLM is stateless and can't remember conversation](https://www.reddit.com/r/Oobabooga/comments/16qa4cj/comment/k1vy8rr/?context=3)***. What actually happens behind the scene is that some api like openAI api takes all your sentences and all GPT-4 responses since the beginning of conversation and merge into 1 big input and let GPT-4 read that 1 big input. Since sequence length is the maximum number of tokens that any LLM can handle where 1 token = 1 word, 1 symbols like colon, it also implies the maximum number of tokens that GPT-4 and all LLM can 'remember'. When we push above LLM maximum sequence length, it gets dumb really quickly as shown in the graph below (high perplexity means dumber LLM)
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/08626f7c-2fa8-47bf-a051-23e8946f4fe7" height="320" width="700"> <br>

2. In theory, we can have infinite sequence length. In practice, the longer the sequence length, the close you get to OOM,the more computation loads is required and memory overheads to be sent back and forth between HBM and SRAM. Contrary to popular believe, the true training/inference time bottleneck isn't computation but memory overheads (see [eli5 FlashAttention-1](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)). Transformer has quadratic complexity between sequence length to computation and memory overheads. FlashAttention-2 can fix some of that, but based Mamba already gotten linear complexity: <br>
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/e87b77bb-b1b9-45c3-8f33-7adb5e926a4e" height="320" width="780"> <br>
from https://github.com/state-spaces/mamba/issues/196 <br>

***If based Mamba is already more efficient than the best Transformer model with FlashAttention-2, then imagine the potential when people starting to optimised Mamba*** <br>
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/2fe1a2d5-cf2b-4d04-8fba-b0ac00d1e881" height="320" width="850"> <br>
from Mamba research paper. Note that memory overhead (not memory storage) is the great bottleneck when it comes to speed <br>


### How does ours differ from other finetuned Mamba?
***We aren't the first ppl trying to finetune mamba. Other had tried before: https://github.com/state-spaces/mamba/pull/83, https://github.com/havenhq/mamba-chat, https://github.com/geronimi73/mamba/blob/main/finetune.py, Incorporate all lessons learnt all existing githubs and extensively reading model cards, Huggingface script and research papers from Mamba and alignment-handbook, we genuinely believe that we have produced a better finetuned mamba LLM.


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

