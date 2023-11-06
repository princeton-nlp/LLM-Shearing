from collections import defaultdict

import numpy as np
from streaming.base.dataset import StreamingDataset

from llmshearing.datasets.streaming_dataset import (
    TextDynamicStreamingDataset, TextStreamingDataset)

dataset = TextDynamicStreamingDataset(local="/scratch/gpfs/mengzhou/llm_data/version5-uint16/500b_dedup_4k/for_prune",
                                      set_names=["cc", "github", "wiki"], 
                                      proportion=[0.5, 0.25, 0.25],
                                      shuffle=True,
                                      is_uint16=True,
                                      max_seq_len=4096)

validd = TextStreamingDataset(local="/scratch/gpfs/mengzhou/llm_data/version5-uint16/500b_dedup_4k/for_prune",
                              split="eval_merge",
                              shuffle=False,
                              is_uint16=True,
                              max_seq_len=4096,
                              num_canonical_nodes=100)

def test_update_proportion():
    # test update_proportion
    iter_dataset = iter(dataset)
    sets = defaultdict(int)
    for i in range(1000):
        sample = next(iter_dataset)
        sets[sample["set"]] += 1
    print(sets)

    dataset.update_proportion([0.1, 0.2, 0.7])
    sets = defaultdict(int)
    for i in range(1000):
        sample = next(iter_dataset)
        sets[sample["set"]] += 1
    print(sets)

def test_load_state_dict():
    # test load state dict 
    used_sample_ids = []
    for i in range(3):
        if i < 2:
            used_sample_ids.append(np.random.randint(dataset.sample_offset_per_stream[i], dataset.sample_offset_per_stream[i+1], 1000).tolist())
        else:
            used_sample_ids.append(np.random.randint(dataset.sample_offset_per_stream[2], dataset.samples_per_stream.sum(), 1000).tolist())
    obj = {"epoch": 0, 
           "num_canonical_nodes": 128,
           "shuffle_seed": 100,
           "proportion": [0.3, 0.2, 0.5], 
           "used_sample_ids": used_sample_ids}
    dataset.load_state_dict(obj)
    test_update_proportion()

def print_valid_dataset_info():
    print(len(validd)) # needs to be divisible by # nodes x # ranks x # workers x batch size
    valid_iter = iter(validd)
    sets = defaultdict(int)
    for i in range(3500):
        sample = next(valid_iter)
        sets[sample["set"]] += 1
    print(sets)

print_valid_dataset_info()