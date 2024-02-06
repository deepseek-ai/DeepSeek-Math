import os
import argparse
from tqdm import tqdm
from glob import glob
import time
import json
import subprocess

from utils import read_data
from data_processing.process_utils import *

_worker_num = int(os.environ.get('WORLD_SIZE', 1))
_worker_id = int(os.environ.get('RANK', 0))

def markup_question(args, item, language, src, task):
    for i in range(len(item['messages']) - 2, -1, -2):
        if language == 'zh':
            if task == 'cot':
                item['messages'][i]['content'] = f"{item['messages'][i]['content']}\n请通过逐步推理来解答问题，并把最终答案放置于" + "\\boxed{}中。"
            elif task == 'tool':
                item['messages'][i]['content'] = f"{item['messages'][i]['content']}\n请结合自然语言和Python程序语言来解答问题，并把最终答案放置于" + "\\boxed{}中。"
            else:
                pass
        elif language == 'en':
            if task == 'cot':
                item['messages'][i]['content'] = f"{item['messages'][i]['content']}\nPlease reason step by step, and put your final answer within " + "\\boxed{}."
            elif task == 'tool':
                item['messages'][i]['content'] = f"{item['messages'][i]['content']}\nPlease integrate natural language reasoning with programs to solve the problem above, and put your final answer within " + "\\boxed{}."
        else:
            pass
    return item

def do_parallel_sampling(args, task, answer_extraction_fn, eval_fn, input_dir, output_dir, log_dir):
    if task == 'pal':
        code_fname = "run_pal_eval"
    elif task == 'cot':
        code_fname = "run_cot_eval"
    elif task == 'tool':
        code_fname = "run_tool_integrated_eval"
    else:
        raise NotImplementedError()

    n_procs = args.ngpus // args.ngpus_per_model

    gpus = [str(i) for i in range(args.ngpus)]
    gpu_groups = []
    for i in range(n_procs):
        gpu_groups.append(gpus[i * args.ngpus_per_model: (i + 1) * args.ngpus_per_model])

    global_n_procs = n_procs * _worker_num

    procs = []
    for pid, gpus in enumerate(gpu_groups):
        global_pid = n_procs * (args.rank or _worker_id) + pid
        logpath = os.path.join(log_dir, f"{global_pid}.log")
        f = open(logpath, "w")
        cmd = f"python infer/{code_fname}.py " \
            f"--data_dir {input_dir} " \
            f"--max_num_examples 100000000000000 " \
            f"--save_dir {output_dir} " \
            f"--model {args.model_path} " \
            f"--tokenizer {args.tokenizer_path or args.model_path} " \
            f"--eval_batch_size 1 " \
            f"--temperature {args.temperature} " \
            f"--repeat_id_start 0 " \
            f"--n_repeat_sampling {args.n_repeats} " \
            f"--n_subsets {global_n_procs} " \
            f"--prompt_format {args.prompt_format} " \
            f"--few_shot_prompt {args.few_shot_prompt} " \
            f"--answer_extraction_fn {answer_extraction_fn} " \
            f"--eval_fn {eval_fn} " \
            f"--subset_id {global_pid} " \
            f"--gpus {','.join(gpus)} "
        if args.use_vllm:
            cmd += " --use_vllm "
        if args.load_in_half:
            cmd += " --load_in_half "
        local_metric_path = os.path.join(output_dir, f"metrics.{global_pid}.json")
        if not args.overwrite and os.path.exists(local_metric_path) and read_data(local_metric_path)['n_samples'] > 0:
            continue
        procs.append((global_pid, subprocess.Popen(cmd.split(), stdout=f, stderr=f), f))
    for (global_pid, proc, f) in procs:
        print(f"Waiting for the {global_pid}th process to finish ...", flush=True)
        proc.wait()
    for (global_pid, proc, f) in procs:
        print(f"Closing the {global_pid}th process ...", flush=True)
        f.close()

    time.sleep(1)

    local_pids = [global_pid for (global_pid, _, _) in procs]

    agg_preds = []
    for fname in glob(os.path.join(output_dir, "predictions.*.json")):
        if any(str(pid) in fname for pid in local_pids):
            agg_preds.extend(read_data(fname))

    metrics = {}
    n_samples = 0
    for fname in glob(os.path.join(output_dir, "metrics.*.json")):
        if not any(str(pid) in fname for pid in local_pids):
            continue
        _metrics = read_data(fname)
        n_samples += _metrics['n_samples']
        for key, val in _metrics.items():
            if key != 'n_samples':
                metrics[key] = metrics.get(key, 0) + val * _metrics['n_samples']
    for key, val in metrics.items():
        metrics[key] = val / max(n_samples, 1)

    result_msg = f"n samples = {n_samples}"
    for key, val in metrics.items():
        result_msg += f"\n{key} = {val * 100}"

    metrics['n_samples'] = n_samples

    return metrics, agg_preds, result_msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True, help="default to `model_path`_predictions")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--model-size", type=str, choices=['1b', '7b', '13b', '33b', '34b', '70b'], default="7b")

    parser.add_argument("--test-conf", type=str, default="configs/zero_shot_test_configs.json", help="path to testing data config file that maps from a source to its info")
    parser.add_argument("--ngpus", type=int, default=8)
    parser.add_argument("--overwrite", action='store_true')
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--use-vllm", action='store_true')
    parser.add_argument("--load_in_half", action='store_true')

    parser.add_argument("--prompt_format", type=str, default="sft")
    parser.add_argument("--few_shot_prompt", type=str, default=None)

    parser.add_argument("--no-markup-question", action='store_true')

    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    print(f"Evaluating {args.model_path}", flush=True)

    if args.output_dir is None:
        args.output_dir = f"{args.model_path.rstrip('/')}_predictions"

    args.ngpus_per_model = 4 if args.model_size in ['70b', '33b', '34b'] else 1
    assert args.ngpus % args.ngpus_per_model == 0

    default_few_shot_prompt = args.few_shot_prompt

    test_conf = read_data(args.test_conf)

    for src, info in test_conf.items():
        if args.n_repeats > 1:
            _src = f"{src}/sample_logs"
        else:
            _src = f"{src}/infer_logs"
        if _worker_num > 1:
            _src = f"{_src}/{args.rank or _worker_id}"
        if args.prompt_format == 'few_shot':
            args.few_shot_prompt = info.get('few_shot_prompt', None) or default_few_shot_prompt
        for task in info['tasks']:
            fname = os.path.join(args.output_dir, _src, task, "test_data", "test.jsonl")
            input_dir = os.path.dirname(fname)
            os.makedirs(input_dir, exist_ok=True)
            metric_path = os.path.join(args.output_dir, _src, task, "samples", "metrics.json")
            if not args.overwrite and os.path.exists(metric_path) and read_data(metric_path)['n_samples'] > 0:
                continue
            with open(fname, "w") as file:
                data = read_data(info['test_path'])
                for i, sample in enumerate(tqdm(data, desc=f'processing {src}')):
                    fn = eval(info['process_fn'])
                    sample['id'] = sample.get('id', f"{src}-{i}")
                    for j, item in enumerate(fn(sample)):
                        item['dataset'] = src
                        item['id'] = f"{src}-test-{i}-{j}"
                        assert 'answer' in item
                        if not args.no_markup_question:
                            item = markup_question(args, item, info['language'], src, task)
                        print(json.dumps(item), file=file, flush=True)

            output_dir = os.path.join(args.output_dir, _src, task, "samples")
            log_dir = os.path.join(args.output_dir, _src, task, "logs")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            metrics, agg_preds, result_msg = do_parallel_sampling(args, task, info['answer_extraction_fn'], info['eval_fn'], input_dir, output_dir, log_dir)

            os.makedirs(os.path.dirname(metric_path), exist_ok=True)
            json.dump(metrics, open(metric_path, "w"), indent=4)
            data_path = os.path.join(args.output_dir, _src, task, "samples", "predictions.json")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            with open(data_path, "w") as file:
                json.dump(agg_preds, file, ensure_ascii=False)
            print(f"src = {src} | task = {task} >>>\n{result_msg}\n\n", flush=True)

if __name__ == '__main__':
    main()
