

import os
import json
import torch
import datasets
import pandas as pd
import transformers

from . import logging, utils
from .data_preprocessor import (
    DataCollatorForSFTDataset,
    DataCollatorForSFTDataset,
    SFTDataset,
    split_train_into_train_and_eval,
)

logger = logging.get_logger(__name__)

def make_ShareGPTsupervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    with open(os.path.join(data_args.dataset_path, data_args.dataset_name + ".json"), "r") as fin :
        alpaca_instructions = json.load(fin)
    train_df = pd.DataFrame(alpaca_instructions)

    train_dataset = SFTDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        source="ShareGPT_SFT",
    )
    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=training_args.seed,
    )
    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)