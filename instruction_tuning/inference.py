import pathlib
import sys
from typing import Dict, Optional, Sequence, Union

import os
import json
import datasets
import fire
import pandas as pd

from alpaca_farm import data_preprocessor, distributed_utils, utils
from alpaca_farm.inference import decode
from alpaca_farm.types import AnyPath, AnyPathOrNone

sample_mode_formatter = "temperature={temperature},max_new_tokens={max_new_tokens},seed={seed}"


def run_decode(
    decoder_name_or_path: AnyPath,
    dataset_path=None,
    dataset_name: Optional[str] = None,
    split="eval",
    prompt_dict_path=None,
    output_path: AnyPathOrNone = None,
    max_instances=sys.maxsize,
    per_device_batch_size=4,
    temperature=1.0,
    max_new_tokens=300,
    num_return_sequences=4,
    mixed_precision=None,
    tf32=False,
    seed: Optional[int] = None,
):
    """Decode samples from the policy language model.

    Args:
        decoder_name_or_path: Name or path of the policy language model.
        dataset_path: Path to the dataset for datasets.load_dataset.
        dataset_name: Name of the dataset for datasets.load_dataset.
        prompt_dict_path: Path to the prompt dictionary for formatting the instruction and input into a string.
        output_path: Optional path to save the decoding results.
        split: Split of the dataset to decode.
        max_instances: Maximum number of instances to decode.
        per_device_batch_size: Batch size for reranking for each device.
        temperature: Temperature for decoding.
        max_new_tokens: Maximum number of new tokens to generate.
        seed: Random seed for decoding.
        num_return_sequences: Number of sequences to return per each prompt.
        mixed_precision: Mixed precision mode for the reward model.
        tf32: Whether to use tensorfloat32 for matrix multiplication.

    Returns:
        List of dict data with keys.
        If num_return_sequences > 1, each 'completion' is a list of strings. Otherwise, it is a string.
    """
    with open(os.path.join(dataset_path, dataset_name + ".json"), "r") as fin :
        dataset = {split : json.load(fin)}
    source = "ShareGPT_SFT"

    prompts, list_dict_data, metadata = data_preprocessor.format_prompt_with_data_frame(
        df=pd.DataFrame(dataset[split]),
        prompt_dict=utils.jload(prompt_dict_path),
        source=source,
    )
    prompts, list_dict_data = prompts[:max_instances], list_dict_data[:max_instances]

    outputs = decode.decode_prompts_with_huggingface(
        model_name_or_path=decoder_name_or_path,
        prompts=prompts,
        decoding_args=decode.HFDecodingArguments(
            temperature=temperature, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences
        ),
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        tf32=tf32,
        seed=seed,
    )

    sample_mode = sample_mode_formatter.format(temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
    return_list_dict_data = [
        {
            "instruction": dict_data["instruction"] if "instrcution" in dict_data else None,
            "input": dict_data["input"],
            "output": output,
            "prompt": prompt,
            "decoder_name_or_path": decoder_name_or_path,
            "sample_mode": sample_mode,
        }
        for dict_data, prompt, output in utils.zip_(list_dict_data, prompts, outputs)
    ]
    if output_path is not None and distributed_utils.is_main_process():
        utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def run_inference(
    decoder_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    prompt_dict_path=None,
    dataset_path=None,
    dataset_name: Optional[str] = None,
    split="eval",
    per_device_batch_size=4,
    max_instances=sys.maxsize,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=300,
    mixed_precision=None,
    tf32=False,
    flash_attn=False,
):
    """Chain together decoding and rerank."""
    decode_return_list_dict_data = run_decode(
        decoder_name_or_path=decoder_name_or_path,
        prompt_dict_path=prompt_dict_path,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        split=split,
        max_instances=max_instances,
        per_device_batch_size=per_device_batch_size,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
        mixed_precision=mixed_precision,
        tf32=tf32,
    )

    if output_path is not None and distributed_utils.is_main_process():
        utils.jdump(decode_return_list_dict_data, os.path.join(decoder_name_or_path, output_path))

    return decode_return_list_dict_data


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
