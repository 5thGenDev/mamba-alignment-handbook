Please see instruction for original alignment-handbook here: https://github.com/huggingface/alignment-handbook/tree/main

### Reproducible pipeline: 
1. Follow instructions of [alignment-handbook git](https://github.com/huggingface/alignment-handbook) and [Mamba git](https://github.com/state-spaces/mamba) to install all dependencies for both on handbook conda environment. To get compatible docker image for Mamba, take nvcr.io/nvidia/pytorch:23.12-py3
2. <git clone https://github.com/5thGenDev/mamba-finetune/tree/main> and <git clone https://github.com/5thGenDev/mamba-alignment-handbook>
3. Before submitting .submit_file for condor to run  any bash script, type "chmod 777 bash script file" to make them executable
If you want to intuitive read our code, read these files that you can see main page: dpo_sft.submit_file, mamba-sft-lora.yaml, mamba.sh, mamba.submit_file, sft.sh. Our yaml is filed based on - https://github.com/huggingface/alignment-handbook/pull/88 with some minor adjustment according to [Mamba model card](https://huggingface.co/state-spaces/mamba-2.8b-hf) and [Mamba paper section E2](https://arxiv.org/pdf/2312.00752.pdf)
   

### Overall pipeline: 
1. Status: Done - Install necessary packages and pull docker image to finetune pretrained state-spaces/mamba-2.8b-hf. 
2. Status: Done - Tweak Mamba git so that class Mambaconfig is more friendly to Huggingface/Transformers api - but not sure if this matters because we're gonna pull transformer-compatible pretrained Mamba anyway.
3. From [Zephyr-7b-beta script](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-beta/README.md), tweak [SFT-qLORA.yaml](https://github.com/5thGenDev/mamba-alignment-handbook/blob/main/mamba-sft-qlora.yaml); then tweak [DPO-qLORA.yaml](https://github.com/5thGenDev/mamba-alignment-handbook/blob/main/mamba-dpo-qlora.yaml) so that they fit Mamba.
4. Right before evaluation, [load sft adapter first then merge the adapter into the base model and then load the dpo adapter](https://github.com/huggingface/alignment-handbook/issues/78).
5. Evaluate on MTBench and AlpacaEval according to [alignment-handbook paper at section 4.2](https://arxiv.org/pdf/2310.16944.pdf).
6. Take finetuned Mamba and repeat (3)->(5) for different datasets used in [Zephyr 7B Gemma scripts](https://github.com/huggingface/alignment-handbook/blob/main/recipes/zephyr-7b-gemma/README.md). Since those are 10k and 7k datasets, even without qLORA, the training time won't be that bad.
To access A100 GPUs, all finetunings were done HPC Condor so I was in charge for (1) and writing Slurm scripts in .submit_file for (2) and (3). Meanwhile everyone including me read across many Git issues on Mamba and DPO-SFT git to tweak parameters on yaml files to adapt SFT-qLORA, DPO-qLORA on Mamba.

### Why mamba?
TLDR: There are rumours within LLM research community at large that Mamba can replace Transformer because ***[it has Linear Complexity when it comes to training time with respect to sequence length](https://github.com/state-spaces/mamba/issues/196), thus it can handle bigger sequence length and is 20-40x faster in training time***. In the graph below produced by Mamba authors, they showcased that comparing purely by network architecture performance (so not taking in account of speedup from better training recipe, more quality dataset,... etc etc), based Mamba with little optimisation in its infant state can beat a Transformer architecture which had 7 years of R&D behind it
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/2fe1a2d5-cf2b-4d04-8fba-b0ac00d1e881" height="320" width="850"> <br>
***Now, let's tackle the impact of bigger sequence length and "reasonable" training time.***

### Why not mamba?
- Since Mamba is in its infant state in R&D right now. It won't receive x2 speed boost from [unsloth](https://huggingface.co/docs/trl/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth), x6 speed boost from [FlashAttention-2](https://huggingface.co/docs/trl/en/sft_trainer#using-flash-attention-2). Luckily, 20x-40x speedboost from Mamba is more than combined x12 speedboost from FlashAttention-2 and Unsloth.
- Not sure if [QLora which worked for Transformer](https://github.com/huggingface/alignment-handbook/pull/88) will also work for Mamba, but that's why we are here - to test and share. However, we have good confidence that there will definitely something like a FlashAttention speed boost for Mamba since the author of Mamba paper: Tri Dao is the same guy in FlashAttention-1 and 2.
- Mamba-2.8b-slimpj (better mamba) is not yet compatible with HuggingFace/Transformers. I've made some editions to try making it compatible but it's not worth it to gamble it now.
   
#### Bigger sequence length
***The reason why "sequence length" is top 3 buzzword whenever there's a new Transformer or Attention paper [is because all LLM stateless and can't remember conversation](https://www.reddit.com/r/Oobabooga/comments/16qa4cj/comment/k1vy8rr/?context=3); so by the time you send it the next prompt within the same chat, it already forgotten who you are***. What actually happens behind the scene is that an api (whether it's Azure, OpenAI, HuggingFace,...) takes all your sentences and all LLM responses since the beginning of conversation and merge into 1 big gigantic prompt and let LLM read that - the technical term for gigantic prompt is "context length". ***Essentially, think of LLM as the girl in that movie "50 first dates" but super smart and can finish reading a prompt as thick as a short story in miliseconds or less***. This calls back to "sequence length" where 
["sequence length" = "context length" + "max prompt"](https://www.reddit.com/r/OpenAI/comments/1329q4a/comment/ji3ynr3/?utm_source=share&utm_medium=web2x&context=3), and sequence length per se is the maximum number of tokens that we can input to an LLM. Right now, both Microsoft and OpenAI is trying to rollout their next LLM with 32k sequence length so you know "sequence length". Anyhow, does bigger sequence length only means you can talk to LLM longer or sending a bigger prompt to GPT-4? Yes, but there's a bigger implication behind it. If you try to talk to any pretrained LLM on HuggingFace, you will realise that it will be dumb initially - here it is an [example](https://github.com/huggingface/alignment-handbook/issues/78). I don't know how to explain this phenomenon, I just know that all LLMs suffer from this phenomenon for the first 500 tokens (oughly 375 words: 500 * [3/4](https://platform.openai.com/tokenizer) = 375) then it gets smart until we get over its maximum "sequence length" as showcased in the graph below (high perplexity means dumber LLM)
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/08626f7c-2fa8-47bf-a051-23e8946f4fe7" height="320" width="620"> <br>
Right now OpenAI and Microsoft are looking to train LLM with 32k sequence length. When it rolls out, finetuning will be bottleneck at [I/O memory overhead bandwidth-bound](https://horace.io/brrr_intro.html) more than ever. From current trend of GPU development, Nvidia’s FLOPS have increased multiple orders of magnitude, primarily architectural changes such as the tensor core and lower precision floating point formats, [whereas memory has not followed the same path](https://www.semianalysis.com/p/cxl-enables-microsoft-azure-to-cut). Even right now, we are already feeling that issue, hence when FlashAttention-1, it was instantly in-demand.

#### 20-40x faster in training time
In theory, we can have infinite sequence length. In practice, longer sequence length means longer training time and bigger price tag. Also, to make a successful model, expect to pretrain or finetune at least several times before you get the parameters just right. 
- Per pretraining, this is the typical price tag: <br>
<img src="https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/1c477925-9db5-414f-b1d9-b8fa70dc9ceb" height="380" width="480"> <br>
from https://www.databricks.com/blog/gpt-3-quality-for-500k where each model was trained on 256xA100-40GB cluster with 1600Gbps RoCE interconnect, using a global batch size of 2048 sequences ~= 4M tokens. <br>

- Per finetuning, if you use 16x A100-80GB, it typically takes 2-4 hours as reported in [Zephyr paper: (SFT + DPO) training recipe](https://arxiv.org/abs/2310.16944); thus for 8x A100, it will take 4-8 hours. With a price tag from [vast.ai](https://cloud.vast.ai/create/?_gl=1*1ohiye0*_ga*NjAwNzg5OTQuMTcwNjk5NDQzMg..*_ga_DG15WC8WXG*MTcxMDE1MDU0My42LjEuMTcxMDE1NDg5MS42MC4wLjA.*_gcl_au*NjQzNDQ2NzU5LjE3MDY5OTQ0MzI.) (cheaper than Google), you are looking at £46.33 -> £92.66. <br>

Now let's imagine we have a working mamba. $2.5M (or £1.95M) is reduced to £48750. £92.66 is reduced down to £2.3. ***Now this is even more important than our project, research trend is gearing toward add-on many training recipe together: First it was [(SFT + DPO)](https://arxiv.org/abs/2310.16944), now it is [(SFT + DPO + UNA)](https://huggingface.co/fblgit/una-xaberius-34b-v1beta). If this trend keeps going, it is inevitable that the demand for training time for finetuning is more becoming more important than ever.***

### How does ours differ from other finetuned Mamba?
***We aren't the first ppl trying to finetune mamba. Other had tried before: https://github.com/state-spaces/mamba/pull/83, https://github.com/havenhq/mamba-chat, https://github.com/geronimi73/mamba/blob/main/finetune.py. To make ours work right, we incorporated all lessons learnt from getting wrong training parameters and we tried to finetune according to Zephyr finetunining recipe (the best afaik).

### Why did we fail
TLDR: We have tried to expect these condor issues ahead (hence the project was designed to be direct and simple so we all could start on implementing straightaway)
![image](https://github.com/5thGenDev/mamba-alignment-handbook/assets/44685200/afd3a2ed-ea02-4e57-a896-1bcc8dc7adb9)

But HPC condor frequently has server issues that are outside our controls where:
- Slurm handles things by different queues but if you're not careful, it can leave resources sitting empty 'just in case'. This happened to us in this project.
- Multi-GPU jobs only gets extra priority when a machine (e.g. A100 compute node) has just come back up from draining, for the first few minutes they 'prefer' multi GPU jobs. This is by design because many machines were getting full of single GPU jobs and the multi jobs couldn't get slots so a 'defrag' system that picks a machine to drain every x hours so that multi gpu jobs can run. Likely happened to us too.
- Some people have been potentially abusing the condor job scheduler to reserve compute resources for themselves by bypassing the limitations set on interactive jobs by submitting regular batch jobs with processes that idle but keep the job running so they can use the condor ssh to job feature to connect to their job and use it interactively. Maybe, but not likely.

### To-do list futures:
- Investigate on QLora, the author himself said that QLora-finetuining chuned out better result than even Full-finetuning: https://github.com/huggingface/alignment-handbook/pull/88. However, using QLora means that we need to investigate on sorting out unquantized base model: https://huggingface.co/docs/trl/main/en/dpo_trainer#downsides-to-merging-qlora-before-dpo-approach-2. Luckily, there are scripts for both, we just need to read 🤓

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
