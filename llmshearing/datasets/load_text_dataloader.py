""" Load text dataloader for training and evaluation. """
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import transformers
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers.data.data_collator import _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from llmshearing.datasets.streaming_dataset import (
    TextDynamicStreamingDataset, TextStreamingDataset)


def build_text_dataloader(cfg: DictConfig, device_batch_size: int, dynamic: bool = False, 
                          set_names: str = None, proportion: List[float] = None) -> DataLoader:
    """Builds a text dataloader.

    Args:
        cfg (DictConfig): Configuration dictionary.
        device_batch_size (int): Batch size for one single device.
        dynamic (bool, optional): Whether to use dynamic streaming dataset to load data from each 
        domain dynamically. Defaults to False.
        set_names (str, optional): Name of the dataset. Defaults to None.
        proportion (List[float], optional): Initial proportion of each domain in the dataset. Defaults to None.

    Returns:
        DataLoader: A PyTorch DataLoader object.
    """
    
    if dynamic:
        dataset = TextDynamicStreamingDataset(local=cfg.dataset.local,
                                              max_seq_len=cfg.dataset.max_seq_len,
                                              batch_size=device_batch_size,
                                              shuffle=cfg.dataset.get(
                                                'shuffle', False),
                                              shuffle_seed=cfg.dataset.get(
                                                'shuffle_seed', 9176),
                                              num_canonical_nodes=cfg.dataset.get(
                                                'num_canonical_nodes', 128),
                                              proportion=proportion,
                                              set_names=set_names,
                                              is_uint16=cfg.dataset.get("is_uint16", False))
    else:
        dataset = TextStreamingDataset(
            local=cfg.dataset.local,
            max_seq_len=cfg.dataset.max_seq_len,
            split=cfg.dataset.get('split', None),
            shuffle=cfg.dataset.get('shuffle', False),
            shuffle_seed=cfg.dataset.get('shuffle_seed', 9176),
            num_canonical_nodes=cfg.dataset.get(
                'num_canonical_nodes', 128),
            batch_size=device_batch_size,
            is_uint16=cfg.dataset.get("is_uint16", False))

    tokenizer = AutoTokenizer.from_pretrained(cfg.dataset.tokenizer_name)
    if isinstance(dataset[0], Mapping) and "set" in dataset[0]:
        COLLATE_FN = DataCollatorForLMWithSetName
        collate_fn = COLLATE_FN(
        set_names=set_names,
        tokenizer=tokenizer,
        mlm=False)
    else:
        COLLATE_FN = transformers.DataCollatorForLanguageModeling
        collate_fn = COLLATE_FN(
            tokenizer=tokenizer,
            mlm=False,
        )
    
    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )


@dataclass
class DataCollatorForLMWithSetName(object):
    """ Data collator used for language modeling with set (domain) name. """
    tokenizer: PreTrainedTokenizerBase # dataclass field must have types 
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    set_names: List[str] = None
    mlm: bool = False

    def __call__(self, features, return_tensors=None):
        return self.torch_call(features)

    def __post_init__(self):
        self.set_name_to_id = defaultdict(int)
        self.set_name_to_id.update({name: i for i, name in enumerate(self.set_names)})

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        input_ids = [example["input_ids"] for example in examples]
        batch = {
            "input_ids": _torch_collate_batch(input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
        }

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # pass the domain name of each example 
        batch["set"] = torch.tensor(
            [self.set_name_to_id[example["set"]] for example in examples])
        if "idx" in examples[0]:
            batch["idx"] = torch.tensor([example["idx"] for example in examples])
        return batch
