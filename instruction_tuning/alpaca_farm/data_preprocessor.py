import numpy
import copy
import dataclasses
from typing import Callable, Dict, Optional, Sequence, Union

import einops
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset

from . import constants, logging, torch_ops, utils
from .types import Tensor

logger = logging.get_logger(__name__)


def format_prompt(example: dict, prompt_dict: dict, source="SFT") -> str:
    """Formats a prompt with a prompt_dict formatter.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".

    Returns:
        A formatted prompt string.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    assert (source == "ShareGPT_SFT" or "instruction" in example) and "input" in example, "Internal error: example missing required keys."

    if example["input"] is None or len(example["input"]) == 0:
        formatted_prompt = prompt_dict["prompt_noinputs"].format_map(example)
    else:
        formatted_prompt = prompt_dict["prompt_inputs"].format_map(example)
    
    if source in ("ShareGPT_SFT", ) :
        pass
    else :
        raise NotImplementedError

    return formatted_prompt


def format_output(example: dict, eos_token: Optional[str] = None, output_key="output", source="SFT") -> str:
    if eos_token is None:
        eos_token = ""
    if source in ("ShareGPT_SFT", ) :
        return f"{example[output_key]}{eos_token}"
    else :
        raise NotImplementedError


def format_prompt_with_data_frame(
    df: pd.DataFrame,
    prompt_dict: dict,
    df_postprocessor: Optional[Callable] = None,
    return_dict=False,
    source="ShareGPT_SFT",
):
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    prompts = [format_prompt(example, prompt_dict, source=source) for example in list_dict_data]
    metadata = {"prompt_dict": prompt_dict}

    if return_dict:
        return dict(prompts=prompts, list_dict_data=list_dict_data, metadata=metadata)
    return prompts, list_dict_data, metadata


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings and return the tokenized content as well metadata (e.g., truncation statistics)."""
    padding = getattr(tokenizer, "padding", "max_length")
    return_overflowing_tokens = transformers.__version__ <= "4.26.1"
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=return_overflowing_tokens,
        )
        for text in strings
    ]

    if padding == "max_length":
        input_ids = labels = torch.cat([tokenized.input_ids for tokenized in tokenized_list])
    else:  # "longest"
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    if return_overflowing_tokens:
        input_ids_lens = labels_lens = [
            tokenizer.model_max_length + tokenized.num_truncated_tokens.item() for tokenized in tokenized_list
        ]
        # `num_truncated_tokens` can be negative, if no truncation occurred.
        num_truncated_tokens = sum(max(tokenized.num_truncated_tokens.item(), 0) for tokenized in tokenized_list)
        num_truncated_examples = sum(tokenized.num_truncated_tokens.item() > 0 for tokenized in tokenized_list)
    else:
        logger.warning(
            "You are using a `transformers` version that does not support `return_overflowing_tokens=True`. "
            "The tokenization metadata will not be recorded."
            "In order to see truncation statistics, please downgrade to `transformers<=4.26.1`."
        )
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        num_truncated_tokens = num_truncated_examples = -1

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        tokenization_metadata=dict(
            num_examples=len(tokenized_list),
            num_truncated_tokens=num_truncated_tokens,
            num_truncated_examples=num_truncated_examples,
            input_ids_avg_len=utils.mean(input_ids_lens),
            input_ids_max_len=max(input_ids_lens),
            input_ids_min_len=min(input_ids_lens),
            labels_avg_len=utils.mean(labels_lens),
            labels_max_len=max(labels_lens),
            labels_min_len=min(labels_lens),
            model_max_length=tokenizer.model_max_length,
        ),
    )


def preprocess_for_sft(
    df: pd.DataFrame,
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor=None,
    verbose=True,
    source="SFT",
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Tokenize each example and create the labels.

    Args:
        df: DataFrame containing the data. Must have columns 'instruction', 'input', and 'output'.
        prompt_dict: Dictionary for formatting prompts.
        tokenizer: Tokenizer to use. If None, use the tokenizer for the given model.
        df_postprocessor: Function to apply to the DataFrame before tokenization.
        verbose: Whether to print tokenization metadata.

    Returns:
        A dictionary mapping str to torch.Tensor.
    """
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    if source in ("ShareGPT_SFT", ) :
        sources = [format_prompt(dict_data, prompt_dict, source=source) for dict_data in list_dict_data]
        targets = [format_output(dict_data, eos_token=tokenizer.eos_token, source=source) for dict_data in list_dict_data]
    else :
        raise NotImplementedError

    examples = [s + t for s, t in utils.zip_(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in utils.zip_(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = constants.IGNORE_INDEX  # Input context should not contribute to loss.

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        metadata=dict(),
        tokenization_metadata=examples_tokenized["tokenization_metadata"],
    )
    if verbose:
        logger.warning(f"Tokenization metadata:\n{utils.jdumps(packaged_data['tokenization_metadata'])}")

    return packaged_data


def get_input_labels_tokenized(sources_tokenized, sources, targets, tokenizer) :
    examples = [s + t for s, t in utils.zip_(sources, targets)]
    examples_tokenized = _tokenize_fn(examples, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in utils.zip_(labels, sources_tokenized["input_ids_lens"]) :
        label[:source_len] = constants.IGNORE_INDEX
    return input_ids, labels, examples_tokenized


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(train_dataset: Dataset, eval_size: int, seed: int) -> tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        source = "ShareGPT_SFT",
    ):
        super(SFTDataset, self).__init__()
        data_dict = preprocess_for_sft(
            df=df, prompt_dict=prompt_dict, tokenizer=tokenizer, df_postprocessor=df_postprocessor,
            source=source,
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.metadata = data_dict["metadata"]
        self.tokenization_metadata = data_dict["tokenization_metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclasses.dataclass
class DataCollatorForSFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=constants.IGNORE_INDEX)
        # When sequences are right padded, `attention_mask` is only useful for T5 training.
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )