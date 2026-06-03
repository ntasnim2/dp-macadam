import gc
import json
import os
import random

import numpy as np
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def free_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def save_results(results_to_add, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            results = json.load(f)
    else:
        results = {}

    results.update(results_to_add)

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved. Keys in file: {list(results.keys())}")


def format_sigma(sigma):
    return str(sigma).replace(".", "_")

