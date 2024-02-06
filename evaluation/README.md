## 1. Introduction

We provide a test script for both zero-shot and few-shot evaluation on mathematical reasoning benchmarks used in our paper.

## 2. Setup

First configure the `prefix` in `environment.yml` and then run the following command
```
conda env create -f environment.yml
```

## 3. Evaluation

For chain-of-thought evaluation of DeepSeekMath-Instruct and DeepSeekMath-RL, our script (see `def markup_question()` in `run_subset_parallel.py`) processes each question as follows:
* English questions: `{question}\nPlease reason step by step, and put your final answer within \\boxed{}.`
* Chinese questions: `{question}\n请通过逐步推理来解答问题，并把最终答案放置于\\boxed{}中。`

For tool-integrated reasoning, we process each question as follows:
* English questions: `{question}\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.`
* Chinese questions: `{question}\n请结合自然语言和Python程序语言来解答问题，并把最终答案放置于\\boxed{}中。`

We provide an example of testing the DeepSeekMath-Base 7B using 8 GPUs.

If you wish to use a different model or dataset, you can modify the configs in `submit_eval_jobs.py` and `configs/*test_configs.json`

```
python submit_eval_jobs.py --n-gpus 8
```

Wait for all processes to finish, and then run the following command to aggregate results from all processes

```
python summarize_results.py [--eval-atp]
```
where the option `--eval-atp` will invoke `unsafe_score_minif2f_isabelle.py` to evaluate the informal-to-formal proving results. Please make sure you have set up the [PISA](https://github.com/wellecks/lm-evaluation-harness/blob/minif2f-isabelle/docs/isabelle_setup.md) server before using this option.

A summary of all evaluation results will be saved as `evaluation_results.json`

## 4. Model Outputs

We provide all model outputs in `outputs.zip`.
