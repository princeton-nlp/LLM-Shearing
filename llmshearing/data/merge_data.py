
import argparse
import contextlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from datasets import load_dataset
from streaming.base.format.mds import MDSReader, MDSWriter
from transformers import LlamaTokenizerFast

from llmshearing.datasets.streaming_dataset import TextStreamingDataset


def load_data(data_local, data_split, tokenizer_name):
    """ load data from a split """
    return TextStreamingDataset(data_local, max_seq_len=2048, split=data_split)

def merge_splits_with_no_bias():
    """ merge splits into one folder without bias, fully read and write, assumes the splits are in the same folder. """
    """ load without uint16 """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_split", type=str, default=None)
    parser.add_argument("--split_names", nargs='+', default=[])
    parser.add_argument("--split_rows", nargs='+', default=[])
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args(sys.argv[2:])
    
    if args.input_dir is None:
        args.input_dir = [Path(os.path.dirname(args.split_names[i])) for i in range(len(args.split_names))]
        args.split_names = [os.path.basename(args.split_names[i]) for i in range(len(args.split_names))]
    else: # if input dirs are different, then only pass split_names in 
        args.input_dir = [Path(args.input_dir)] * len(args.split_names)
    
    first_index_file = json.load(open(args.input_dir[0] / args.split_names[0] / "index.json", "r"))
    column_names = first_index_file["shards"][0]["column_names"]
    column_encodings = first_index_file["shards"][0]["column_encodings"]
    columns = {name: encoding for name, encoding in zip(column_names, column_encodings)}
    out = MDSWriter(columns=columns,
                    out=os.path.join(args.output_dir, args.output_split),
                    compression=None)
    all_samples = []
    
    # shuffle
    if args.shuffle:
        for i, (input_dir, split_name) in enumerate(zip(args.input_dir, args.split_names)):
            print("Getting data from split", split_name)
            data = load_data(str(input_dir), split_name)
            if len(args.split_rows) > 0:
                lens = int(args.split_rows[i])
            else:
                lens = len(data)
            for i in range(lens):
                all_samples.append(data.get_sample(i))
        print("Get all the data", len(all_samples))
        print("Shuffle data")
        index = np.random.permutation(len(all_samples))
        for i in index: 
            out.write(all_samples[i])
    else:
        for i, (input_dir, split_name) in enumerate(zip(args.input_dir, args.split_names)):
            data = load_data(str(input_dir), split_name)
            if len(args.split_rows) > 0:
                lens = int(args.split_rows[i])
            else:
                lens = len(data)
            for i in range(lens):
                out.write(data.get_sample(i))
            print("Finish writing split", input_dir / split_name)
    out.finish()

if __name__ == "__main__":
    merge_splits_with_no_bias()