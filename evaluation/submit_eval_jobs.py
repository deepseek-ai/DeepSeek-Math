import os
import argparse

configs = [
    {
        'output-dir': "outputs/DeepSeekMath-Base",
        'model-path': "deepseek-ai/deepseek-math-7b-base",
        'tokenizer-path': "deepseek-ai/deepseek-math-7b-base",
        'model-size': "7b",
        'overwrite': False,
        'use-vllm': True,
        'no-markup-question': True,
        'test-conf': "configs/few_shot_test_configs.json",
        'prompt_format': 'few_shot',
        'expname': 'eval-deepseek-math-7b-base'
    },
    {
        'output-dir': "outputs/DeepSeekMath-Instruct",
        'model-path': "deepseek-ai/deepseek-math-7b-instruct",
        'tokenizer-path': "deepseek-ai/deepseek-math-7b-instruct",
        'model-size': "7b",
        'overwrite': False,
        'use-vllm': True,
        'test-conf': "configs/zero_shot_test_configs.json",
        'expname': 'eval-deepseek-math-7b-instruct'
    },
    {
        'output-dir': "outputs/DeepSeekMath-RL",
        'model-path': "deepseek-ai/deepseek-math-7b-rl",
        'tokenizer-path': "deepseek-ai/deepseek-math-7b-rl",
        'model-size': "7b",
        'overwrite': False,
        'use-vllm': True,
        'test-conf': "configs/zero_shot_test_configs.json",
        'expname': 'eval-deepseek-math-7b-rl'
    }
]

base_conf, instruct_conf, rl_conf = configs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int ,default=1)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--n-gpus", type=int, default=8)
    args = parser.parse_args()

    conf = base_conf # TODO: your conf here
    cmd = "python run_subset_parallel.py"
    for key, val in conf.items():
        if key == 'expname':
            continue
        if isinstance(val, str):
            cmd += f" --{key} {val}"
        elif val:
            cmd += f" --{key}"
    cmd += f" --test-conf {conf['test-conf']}"
    cmd += f" --n-repeats {args.n_repeats}"
    cmd += f" --temperature {args.temperature}"
    cmd += f" --ngpus {args.n_gpus}"
    cmd += f" --rank {0} &"
    print(cmd, flush=True)
    os.system(cmd)

if __name__ == '__main__':
    main()
