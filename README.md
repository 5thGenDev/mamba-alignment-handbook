Please see instruction for original alignment-handbook here: https://github.com/huggingface/alignment-handbook/tree/main


### Reproducible pipeline: 
1. Git clone based alignment-handbook and based mamba and follow their installation instructions to get all necessary dependencies
2. Delete git clone of based alignment-handbook and based mamba
3. Git clone my modified mamba and alignment-handbook: https://github.com/5thGenDev/mamba-finetune/tree/main and https://github.com/5thGenDev/mamba-alignment-handbook. Based mamba can't finetune on downstream task and based-alignment-handbook has different config than mamba config even though they both inherit from HuggingFace, so some modified code is needed.
4. Read the instruction in https://github.com/5thGenDev/mamba-finetune/blob/main/README.md carefully because it tells you which .py to make manual adjustment yourself.
***Expect a lot of bugs at the moment because we couldn't test it on our HPC condor. This script is meant to run on uni condor HPC so no jupyter notebook here, thus before running these bash script, type "chmod 777 mamba.sh or sft.sh" before you submit mamba.submit_file and then dpo_sft.submit_file*** If you want to intuitive read our code, read these files that you can see main page: dpo_sft.submit_file, mamba-sft-lora.yaml, mamba.sh, mamba.submit_file, sft.sh

### Overall pipeline: 
1. Install necessary packages and pull docker image to host pretrained state-spaces/mamba-2.8b-slimpj.
2. Finetune pretrained mamba on SFT and DPO training script according to alignment-handbook github. Specifically these twos: https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-beta, https://github.com/huggingface/alignment-handbook/tree/main/recipes/zephyr-7b-gemma
4. Evaluate on MTBench and AlpacaEval according to the instruction from alignment-handbook research paper: https://arxiv.org/pdf/2310.16944.pdf section 4.2
***We are half stucked on (2) (3) because condor decided not to run my Slurm script (see [Why did we fail section](https://github.com/5thGenDev/mamba-alignment-handbook/blob/main/README.md#why-did-we-fail) below)***. To access A100 GPUs, all finetunings were done HPC Condor so I was in charge for (1) and writing Slurm scripts in .submit_file for (2) and (3). Meanwhile everyone including me investigated across many Git issue posts of Mamba and Zephyr in order to nail down the exact parameters to be tweaked when adapting Mamba onto SFT, DPO finetuning scripts.

### But why mamba?
TLDR: There are rumours within LLM research community at large that Mamba can replace Transformer because ***[it has Linear Complexity when it comes to training time with respect to sequence length](https://github.com/state-spaces/mamba/issues/196), thus it can handle bigger sequence length and is 20-40x faster in training time***. In the graph below produced by Mamba authors, they showcased that comparing purely by network architecture performance (so not taking in account of speedup from better training recipe, more quality dataset,... etc etc), based Mamba with little optimisation in its infant state can beat a Transformer architecture which had 7 years of R&D behind it and a further speed boost from FlashAttention-2.
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/2fe1a2d5-cf2b-4d04-8fba-b0ac00d1e881" height="320" width="850"> <br>
***Now, let's tackle the impact of bigger sequence length and "reasonable" training time.***
   
#### Bigger sequence length
***[There is a reason why "sequence length" is top 3 buzzword whenever there's a new paper claims to optimises LLM - This is because all LLM stateless and can't remember conversation](https://www.reddit.com/r/Oobabooga/comments/16qa4cj/comment/k1vy8rr/?context=3); so by the time you send it the next prompt within the same chat, it already forgotten who you are***. What actually happens behind the scene is that an api (whether it's Azure, OpenAI, HuggingFace,...) takes all your sentences and all LLM responses since the beginning of conversation and merge into 1 big gigantic prompt and let LLM read that - the technical term for gigantic prompt is "context length". ***Essentially, think of LLM as the girl in that movie "50 first dates" but super smart and can finish reading a prompt as thick as a short story in miliseconds or less***. This calls back to "sequence length" where 
["sequence length" = "context length" + "max prompt"](https://www.reddit.com/r/OpenAI/comments/1329q4a/comment/ji3ynr3/?utm_source=share&utm_medium=web2x&context=3), and sequence length per se is the maximum number of tokens that we can input to an LLM. Right now, both Microsoft and OpenAI is trying to rollout their next LLM with 32k sequence length so you know "sequence length". Anyhow, does bigger sequence length only means you can talk to LLM longer or sending a bigger prompt to GPT-4? Yes, but there's a bigger implication behind it. If you try to talk to any pretrained LLM on HuggingFace, you will realise that it will be dumb initially - here it is an [example](https://github.com/huggingface/alignment-handbook/issues/78). I don't know how to explain this phenomenon, I just know that all LLMs suffer from this phenomenon for the first 500 tokens (oughly 375 words: 500 * [3/4](https://platform.openai.com/tokenizer) = 375) then it gets smart until we get over its maximum "sequence length" as showcased in the graph below (high perplexity means dumber LLM)
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/08626f7c-2fa8-47bf-a051-23e8946f4fe7" height="320" width="620"> <br>

#### 20-40x faster in training time
In theory, we can have infinite sequence length. In practice, longer sequence length means longer training time and bigger price tag. Also, to make a successful model, expect to pretrain or finetune at least several times before you get the parameters just right. 
- Per pretraining, this is the typical price tag: <br>
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/1c477925-9db5-414f-b1d9-b8fa70dc9ceb" height="380" width="580"> <br>
from https://www.databricks.com/blog/gpt-3-quality-for-500k where each model was trained on 256xA100-40GB cluster with 1600Gbps RoCE interconnect, using a global batch size of 2048 sequences ~= 4M tokens. <br>

- Per finetuning, if you use 16x A100-80GB, it typically takes 2-4 hours as reported in [Zephyr paper: (SFT + DPO) training recipe](https://arxiv.org/abs/2310.16944); thus for 8x A100, it will take 4-8 hours. With a price tag from [vast.ai](https://cloud.vast.ai/create/?_gl=1*1ohiye0*_ga*NjAwNzg5OTQuMTcwNjk5NDQzMg..*_ga_DG15WC8WXG*MTcxMDE1MDU0My42LjEuMTcxMDE1NDg5MS42MC4wLjA.*_gcl_au*NjQzNDQ2NzU5LjE3MDY5OTQ0MzI.) (cheaper than Google), you are looking at £46.33 -> £92.66. <br>

Now let's imagine we have a working mamba. $2.5M (or £1.95M) is reduced to £48750. £92.66 is reduced down to £2.3. ***Now this is even more important than our project, research trend is gearing toward add-on many training recipe together: First it was [(SFT + DPO)](https://arxiv.org/abs/2310.16944), now it is [(SFT + DPO + UNA)](https://huggingface.co/fblgit/una-xaberius-34b-v1beta). If this trend keeps going, with or without LORA, it is inevitable that the demand for training time for finetuning is more becoming more important than ever.***

### How does ours differ from other finetuned Mamba?
***We aren't the first ppl trying to finetune mamba. Other had tried before: https://github.com/state-spaces/mamba/pull/83, https://github.com/havenhq/mamba-chat, https://github.com/geronimi73/mamba/blob/main/finetune.py. To make ours work right, we incorporated all lessons learnt from getting wrong training parameters and we tried to finetune according to Zephyr finetunining recipe (the best afaik).

### Why did we fail
TLDR: We have tried to expect these condor issues ahead (hence the project was designed to be direct and simple so we all could start on implementing straightaway)
![image](https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/afd3a2ed-ea02-4e57-a896-1bcc8dc7adb9)

But HPC condor frequently has server issues that are outside our controls where:
- Slurm handles things by different queues but if you're not careful, it can leave resources sitting empty 'just in case'. This happened to us in this project.
- Multi-GPU jobs only gets extra priority when a machine (e.g. A100 compute node) has just come back up from draining, for the first few minutes they 'prefer' multi GPU jobs. This is by design because many machines were getting full of single GPU jobs and the multi jobs couldn't get slots so a 'defrag' system that picks a machine to drain every x hours so that multi gpu jobs can run. Likely happened to us too.
- Some people have been potentially abusing the condor job scheduler to reserve compute resources for themselves by bypassing the limitations set on interactive jobs by submitting regular batch jobs with processes that idle but keep the job running so they can use the condor ssh to job feature to connect to their job and use it interactively. Maybe, but not likely.


#### What does a token look like
A token can be a word or symbol like , ; . / @
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/5be21d1d-382d-486e-84c3-46a95c74865b" height="350" width="850"> <br>
Credit to [Ark video about Intuition behind Transformers which he went extra steps to explain about tokenization](https://youtu.be/g2BRIuln4uc?t=343) <br>

### Authors
- Nam Tran: namhoangtran1590@gmail.com
- Yousself El Aasar: joeace2002@gmail.com
- Govind K: govindkonnanat2002@gmail.com
- Devoprasad Sunil Nedungade: nedungade@gmail.com
- Sushen YADAV: Sushenydv@gmail.com
