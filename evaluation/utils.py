import json
import random
import numpy as np

def set_seed(seed):
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)

def shuffle(data, seed):
    if seed < 0:
        return data
    set_seed(seed)
    indices = list(range(len(data)))
    np.random.shuffle(indices)
    data = [data[i] for i in indices]
    return data

def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data
