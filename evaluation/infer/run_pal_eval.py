import argparse
import os
import sys
import_path = os.path.abspath(__file__)
for _ in range(2):
    import_path = os.path.dirname(import_path)
sys.path.append(import_path)

import json
from copy import deepcopy
from functools import partial
from vllm import LLM, SamplingParams
from pebble import ProcessPool
from concurrent.futures import TimeoutError
import random
from eval.utils import generate_completions, load_hf_lm_and_tokenizer
from eval.python_executor import PythonExecutor

from transformers import AutoTokenizer

from data_processing.answer_extraction import *
from eval.eval_script import *
from few_shot_prompts import *

def evaluate(eval_fn, tasks, _timeout=15):
    with ProcessPool() as pool:
        timeout_cnt = 0
        iterator = pool.map(eval_fn, tasks, timeout=_timeout).result()
        labels = []
        while True:
            try:
                labels.append(int(next(iterator)))
            except StopIteration:
                break
            except TimeoutError as error:
                labels.append(0)
                timeout_cnt += 1
            except Exception as error:
                print(error.traceback, flush=True)
                exit()
    return labels, timeout_cnt

def main(args):
    random.seed(42)

    print("Loading data...")
    test_data = []
    with open(os.path.join(args.data_dir, f"train.jsonl" if args.infer_train_set else f"test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            messages = example['messages']
            assert len(messages) in [2, 3]
            assert messages[-1]['role'] == 'assistant'
            if not args.complete_partial_output:
                example['reference'] = example.get('reference', '') or messages[-1]['content']
                messages[-1]['content'] = ''
            example['messages'] = messages
            test_data.append(example)

    if args.max_num_examples and len(test_data) > args.max_num_examples:
        test_data = random.sample(test_data, args.max_num_examples)

    if args.n_subsets > 1:
        assert args.subset_id >= 0 and args.subset_id < args.n_subsets
        test_data = [item for i, item in enumerate(test_data) if i % args.n_subsets == args.subset_id]

    if not test_data:
        return

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    if args.prompt_format == 'few_shot':
        assert args.few_shot_prompt is not None
        prompting = eval(args.few_shot_prompt)()

    prompts = []
    for example in test_data:
        prompt = ""
        if args.prompt_format == 'few_shot':
            prompt = prompting.format_prompt(example['messages'][-2]['content'], example['messages'][-1]['content'])
        else:
            for mess in example['messages']:
                if args.prompt_format == 'sft':
                    if mess['role'] == 'user':
                        prompt += f"User: {mess['content'].strip()}\n\nAssistant:"
                    elif mess['role'] == 'assistant':
                        prompt += mess['content'].strip()
                else:
                    raise NotImplementedError()
            prompt = prompt.lstrip()
        example['prompt'] = prompt
        prompts.append(prompt.lstrip())

    global model, tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    print("Loading model and tokenizer...")
    if args.use_vllm:
        if model is None:
            model = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer_name_or_path, trust_remote_code=True, tensor_parallel_size=len(os.environ['CUDA_VISIBLE_DEVICES'].split(",")))
        eos_token = tokenizer.eos_token if tokenizer is not None and tokenizer.eos_token is not None else '</s>'
        stop_words = [eos_token]
        if args.prompt_format == 'few_shot':
            stop_words.extend(prompting.stop_words())
        outputs = model.generate(prompts, SamplingParams(temperature=args.temperature, top_p=1.0, max_tokens=1024, n=1, stop=stop_words))
        outputs = sorted(outputs, key=lambda x: int(x.request_id)) # sort outputs by request_id
        outputs = [output.outputs[0].text for output in outputs]
    else:
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path, 
            load_in_8bit=args.load_in_8bit, 
            load_in_half=args.load_in_half,
            gptq_model=args.gptq
        )

        stop_id_sequences = []
        if tokenizer.eos_token_id is not None:
            stop_id_sequences = [[tokenizer.eos_token_id]]
        if args.prompt_format == 'few_shot':
            stop_id_sequences.extend([tokenizer.encode(word) for word in prompting.stop_words()])
        outputs, finish_completion = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=512,
            batch_size=args.eval_batch_size,
            stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
            end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
        )

    if args.complete_partial_output:
        model_outputs = [example['messages'][-1]['content'] + output for example, output in zip(test_data, outputs)]
    else:
        model_outputs = outputs

    if 'PALGSMPrompt' in args.few_shot_prompt:
        executor = PythonExecutor(get_answer_expr='solution()')
        codes = model_outputs
    elif 'PALMathPrompt' in args.few_shot_prompt:
        executor = PythonExecutor(get_answer_symbol='answer')
        codes = []
        for text in model_outputs:
            if text.count("```") == 4:
                segments = text.split("```")
                assert len(segments) == 5
                code = f"{segments[3]}\n\n{segments[1]}"
            else:
                code = "answer = '[invalid]'"
            codes.append(code)
    else:
        raise NotImplementedError()

    predictions = []
    runtime_errors = []
    for pred, err in executor.batch_apply(codes):
        predictions.append(str(pred))
        runtime_errors.append(str(err['exec_info']).strip())

    assert len(model_outputs) > 0, f"{len(model_outputs)}"

    results = []
    for example, output, pred in zip(test_data, model_outputs, predictions):
        item = deepcopy(example)
        item.update({
            'model_output': output,
            'program_output': pred,
        })
        results.append(item)

    labels, eval_timeout_cnt = evaluate(partial(eval(args.eval_fn), pred_key='program_output'), results)
    for item, label in zip(results, labels):
        item['accuracy'] = label

    print("Calculating accuracy...")
    acc = 0
    for item in results:
        acc += item['accuracy']
    print("output acc = {:.5f}".format(acc / len(results) * 100), flush=True)

    print(f"Timeout count >>> output eval = {eval_timeout_cnt}", flush=True)

    pred_fname = "predictions.json"
    if args.n_subsets > 1:
        pred_fname = f"predictions.{args.subset_id}.json"
    with open(os.path.join(args.save_dir, pred_fname), "w") as fout:
        json.dump(results, fout, ensure_ascii=True)

    metric_fname = "metrics.json"
    if args.n_subsets > 1:
        metric_fname = f"metrics.{args.subset_id}.json"
    with open(os.path.join(args.save_dir, metric_fname), "w") as fout:
        json.dump({
            "n_samples": len(results),
            "accuracy": sum(item['accuracy'] for item in results) / len(results),
        }, fout, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/mgsm")
    parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--save_dir", type=str, default="results/mgsm")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true", help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--load_in_half", action='store_true')
    parser.add_argument("--infer_train_set", action="store_true")
    parser.add_argument("--n_subsets", type=int, default=1)
    parser.add_argument("--subset_id", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeat_id_start", type=int, default=0)
    parser.add_argument("--n_repeat_sampling", type=int, default=1)
    parser.add_argument("--complete_partial_output", action='store_true')
    parser.add_argument("--prompt_format", type=str, choices=['sft', 'few_shot'], default='sft')
    parser.add_argument("--few_shot_prompt", type=str, default=None)
    parser.add_argument("--answer_extraction_fn", type=str, default=None)
    parser.add_argument("--eval_fn", type=str, required=True)
    parser.add_argument("--gpus", type=str, default=None)
    args, unparsed_args = parser.parse_known_args()
    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    print(unparsed_args, flush=True)

    model = None
    tokenizer = None
    pool = None
    if args.n_repeat_sampling > 1 or args.repeat_id_start != 0:
        assert args.temperature > 0
        save_dir = args.save_dir
        for i in range(args.repeat_id_start, args.repeat_id_start + args.n_repeat_sampling):
            print(f"working on the {i} trials ...", flush=True)
            args.save_dir = os.path.join(save_dir, str(i))
            os.makedirs(args.save_dir, exist_ok=True)
            main(args)
    else:
        main(args)

    if pool is not None:
        pool.close()
