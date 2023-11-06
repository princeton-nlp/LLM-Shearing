import json
import random
import numpy as np
from streaming import MDSWriter
import os
from tqdm import tqdm
import argparse

def make_dir_if_not_ex(path):
    if not os.path.exists(path):
        print("Make target folder:", path)
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument("--tokenized_dir", type=str, help="Target directory to save tokenized numpy")
parser.add_argument("--target_dir", type=str, help="Target directory to save tokenized numpy")
parser.add_argument("--eval_seq", type=int, default=2, help="How many sequences to sample for eval for each domain")
parser.add_argument("--seq_length", type=int, default=4096, help="Sequence length")
parser.add_argument("--for_prune", type=float, default=0.001, help="How many tokens (billion) sampled for pruning")
parser.add_argument("--for_ft", type=float, default=0.001, help="How many tokens (billion) sampled for ft")

args = parser.parse_args()

target_folder = args.target_dir
index_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

# RedPajama sampling rate
folders = {
    "arxiv": 0.025, 
    "book": 0.045, 
    "c4-rp": 0.15, 
    "cc": 0.67,
    "github": 0.045, 
    "stackexchange": 0.02, 
    "wiki": 0.045
}
files = open("jsonl_list.txt").readlines()
folder_to_files = {f: [] for f in folders}
for line in files:
    tname = os.path.join(args.tokenized_dir, os.path.splitext(line)[0] + ".npy")
    for split in folders:
        if line[:len(split)] == split:
            folder_to_files[split].append(tname)

target_folders = [list(folders.keys())[index_id]] 

random.seed(42)
np.random.seed(42)

# Eval first
print("Sampling eval data...")
for folder in target_folders:
    print("Split: %s" % folder)
    random.shuffle(folder_to_files[folder])
    # Use the first half of files as evaluation
    selected = folder_to_files[folder][:len(folder_to_files[folder]) // 2]
    # The left will be used for training later
    folder_to_files[folder] = folder_to_files[folder][len(folder_to_files[folder]) // 2:]

    # Check if the files exist    
    selected_verified = []
    for fname in selected:
        if os.path.exists(fname):
            selected_verified.append(fname)
    selected = selected_verified
    if len(selected) == 0:
        import pdb; pdb.set_trace()

    folder_eval_target = args.eval_seq
    num_sample_each_file = max(1, folder_eval_target // len(selected) + 1)
    print("  sample from %d files, %d samples each, total %d" % (len(selected), num_sample_each_file, folder_eval_target))
    out = MDSWriter(
        columns={"tokens": "bytes", "set": "str"}, 
        out=os.path.join(target_folder, "eval", folder), 
        compression=None
    )
    total = 0
    for fname in tqdm(selected):
        data = np.load(fname)
        if len(data) % 2 != 0:
            data = data[:-1]
        data = data.reshape(-1, args.seq_length)

        indices = np.random.choice(len(data), num_sample_each_file, replace=False)
        sampled_data = data[indices]
        for sample in sampled_data:
            out.write({
                "tokens": sample.tobytes(),
                "set": folder
            })
            total += 1
            if total >= folder_eval_target:
                break
        if total >= folder_eval_target:
            print("Hit eval target")
            break
    out.finish()
    print("Total: %d" % total)

print("Eval done.")

# Train then
seq_1b = 1000000000 // args.seq_length # this leads to roughly 1B data 

for_prune = int(seq_1b * args.for_prune) # #seq for prune
for_ft = int(seq_1b * args.for_ft) # #seq for ft

print("Sampling pruning data...")
for folder in target_folders:
    print("Split: %s" % folder)
    random.shuffle(folder_to_files[folder])
    # This is what was left after sampling eval
    selected = folder_to_files[folder]

    # Check if the files exist    
    selected_verified = []
    for fname in selected:
        if os.path.exists(fname):
            selected_verified.append(fname)
    selected = selected_verified
    if len(selected) == 0:
        import pdb; pdb.set_trace()

    folder_for_prune = int(for_prune * folders[folder])
    file_for_prune = max(1, folder_for_prune // len(selected) + 1)

    folder_for_ft = int(for_ft * folders[folder])
    file_for_ft = max(1, folder_for_ft // len(selected) + 1)

    print(f"In total {len(selected)} files")
    print(f"For prune sample {folder_for_prune} in total, {file_for_prune} for each file")
    print(f"For ft sample {folder_for_ft} in total, {file_for_ft} for each file")

    make_dir_if_not_ex(os.path.join(target_folder, "for_prune"))
    make_dir_if_not_ex(os.path.join(target_folder, "for_ft"))

    prune_out = MDSWriter(
        columns={"tokens": "bytes", "set": "str"}, 
        out=os.path.join(target_folder, "for_prune", folder), 
        compression=None
    )
    ft_out = MDSWriter(
        columns={"tokens": "bytes", "set": "str"}, 
        out=os.path.join(target_folder, "for_ft", folder), 
        compression=None
    )


    total_prune = 0
    total_ft = 0

    for fname in tqdm(selected):
        data = np.load(fname)
        if len(data) % 2 != 0:
            data = data[:-1]
        data = data.reshape(-1, args.seq_length)

        indices = np.arange(len(data))
        np.random.shuffle(indices)
        prune_indices = indices[:file_for_prune]
        ft_indices = indices[file_for_prune:file_for_prune+file_for_ft]

        prune_data = data[prune_indices]
        for sample in prune_data:
            prune_out.write({
                "tokens": sample.tobytes(),
                "set": folder
            })
            total_prune += 1
            if total_prune >= folder_for_prune:
                break
        ft_data = data[ft_indices]
        for sample in ft_data:
            ft_out.write({
                "tokens": sample.tobytes(),
                "set": folder
            })
            total_ft += 1
            if total_ft >= folder_for_ft:
                break


        if total_prune >= folder_for_prune and total_ft >= folder_for_ft:
            break

    prune_out.finish()
    ft_out.finish()


print("Done.")
