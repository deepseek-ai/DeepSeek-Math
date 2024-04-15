import os
import json
import argparse

from glob import glob
from copy import deepcopy

def seek_metrics(path):
    if os.path.isdir(path):
        for subpath in glob(os.path.join(path, "*")):
            yield from seek_metrics(subpath)
    else:
        if "metrics.json" in path:
            yield path

def seek_predictions(path):
    if os.path.isdir(path):
        for subpath in glob(os.path.join(path, "*")):
            yield from seek_predictions(subpath)
    else:
        if "predictions.json" in path:
            yield path

def aggregate_metrics(paths):
    result = {}
    total = 0
    for path in paths:
        metric = json.load(open(path, "r"))
        n_samples = metric['n_samples']
        total += n_samples
        for key, val in metric.items():
            if key != 'n_samples':
                result[key] = result.get(key, 0) + val * n_samples
    for key, val in result.items():
        result[key] = val / total
    result['n_samples'] = total
    return result

def aggregate_predictions(paths):
    data = []
    for path in paths:
        try:
            data.extend(json.load(open(path, "r")))
        except:
            print(path, flush=True)
            continue
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type=str, default="outputs")
    parser.add_argument("--eval-atp", action='store_true')
    parser.add_argument("--isa-path", type=str, default="")
    parser.add_argument("--theory-file", type=str, default="")
    args = parser.parse_args()

    model2dataset2task2metric = {}
    for model in os.listdir(args.dirname):
        model2dataset2task2metric[model] = {}
        subdir = os.path.join(args.dirname, model)
        for dataset in os.listdir(subdir):
            log_dir = os.path.join(subdir, dataset, "infer_logs")
            agg_dirname = os.path.join(subdir, dataset, "results")
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
                os.system(f"mv {subdir}/{dataset}/* {log_dir}")
            metric_paths = list(seek_metrics(log_dir))
            pred_paths = list(seek_predictions(log_dir))
            task2metric_paths = {'cot': [], 'tool': []}
            task2pred_paths = {'cot': [], 'tool': []}
            for path in metric_paths:
                if 'cot' in path:
                    task2metric_paths['cot'].append(path)
                else:
                    task2metric_paths['tool'].append(path)
            for path in pred_paths:
                if 'cot' in path:
                    task2pred_paths['cot'].append(path)
                else:
                    task2pred_paths['tool'].append(path)
            task2metric = {task: aggregate_metrics(paths) for task, paths in task2metric_paths.items()}
            task2pred = {task: aggregate_predictions(paths) for task, paths in task2pred_paths.items()}
            model2dataset2task2metric[model][dataset] = task2metric

            for task in task2metric:
                task_dirname = os.path.join(agg_dirname, task)
                os.makedirs(task_dirname, exist_ok=True)
                metric_path = os.path.join(task_dirname, "metrics.json")
                pred_path = os.path.join(task_dirname, "predictions.json")
                json.dump(task2metric[task], open(metric_path, "w"), indent=4)
                json.dump(task2pred[task], open(pred_path, "w"), indent=4)
                if 'minif2f' in dataset.lower() and 'isabelle' in dataset.lower() and task2pred[task] and args.eval_atp:
                    eval_path = metric_path + ".eval"
                    if os.path.exists(eval_path) and json.load(open(eval_path, "r")).get('n_samples', 0):
                        model2dataset2task2metric[model][dataset][task] = json.load(open(eval_path, "r"))
                        continue
                    print(f"Running minif2f-isabelle evaluation on {dataset} ...", flush=True)
                    print(f"Predictions >>> {pred_path}", flush=True)
                    cmd = f"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python unsafe_score_minif2f_isabelle.py " \
                        f"--isa-path {args.isa_path} " \
                        f"--theory-file {args.theory_file} " \
                        f"--working-dir {args.working_dir} " \
                        f"--port 9000 " \
                        f"--output {pred_path} "
                    os.system(cmd)

    json.dump(model2dataset2task2metric, open("evaluation_results.json", "w"), indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
