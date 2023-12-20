## Pruning data for reproducibility purposes
We've made the pruning data we used during the entire pruning process available on Google Drive [here](https://drive.google.com/drive/folders/1A_-88BqcOGa1Pbo-1ZShG2saU0cdRZgK). You can access it by clicking here. Alternatively, you have the option to process your own data as follows.


## Data preprocessing

We provide preprocessing code to tokenize, sample, and process RedPajama data into MDS format ([Mosaic's streaming package](https://docs.mosaicml.com/projects/streaming/en/stable/index.html)). Here we have some sampled RedPajama data in `sample_redpajama`, but you should download the original [RedPajama](https://together.ai/blog/redpajama) data (or other data) and organize it in the following way: in your data directory, each folder is a domain and within the folder, there are a series of jsonl files. For the jsonl file format we follow the original RedPajama format.

### Step 1: get all files

Run the following command
```bash
python get_all_jsonl.py {PATH TO YOUR DATA}
```
This will read all the files from the data directory and save all the jsonl file relative paths in `jsonl_list.txt`. Make sure that there are no non-data files in the data folder or make sure to delete them from `jsonl_list.txt` afterwards. In this example, you can use `sample_redpajama` as the data directory.

### Step 2: tokenize all files

We'll then tokenize all files. You can run the example script:
```bash
bash tokenize_all_files.sh
```
and all files will be tokenized one by one and the numpy version will be saved in `tokenized_sample_redpajama`. You can customize the sequence length and source/target directory in the script. In the script, we use `SLURM_ARRAY_TASK_ID` to specify which file from `jsonl_list.txt` to tokenize. This will make it much easier to parallelize if you use a SLURM system: simple run a SLURM array job and all your files will be tokenized in parallel.

### Step 3: sample data for eval/pruning/fine-tuning and encode as MDS

We need to sample the data into eval, pruning, and fine-tuning, and encode them as MDS, the format used by [Mosaic's streaming package](https://docs.mosaicml.com/projects/streaming/en/stable/index.html). To see an example, run
```bash
bash sample_all_domains.sh
```

In this script, you can control the source/target directory, sequence length, and how much data to sample for eval/pruning/fine-tuning respectively. Note that to make sure that there is no eval leakage, **we first split all files in one domain to two disjoint sets, and sample eval from one set and pruning/fine-tuning from others** -- that means you need to have at least two files for each domain. You can also change the domain sample ratio in the script. Similar to tokenization, you can use `SLURM_ARRAY_TASK_ID` to control which domain to work on, and this can be easily parallized by using a SLURM array job. 

**Note: our code will try to sample an equal amount of samples in different files for a diverse distribution. We suggest you split each domain into many equal-size files. If there are too few files or the file sizes vary a lot, it might lead to imbalanced sampling.**

In the end, the target data folder will look like this:

```
eval
| domain1
| domain2
| eval_merge
for_prune
| domain1
| domain2
for_ft
| domain1
| domain2
```

Note that the eval folder must include `eval_merge`, which is a single split that contains validation data of all the domains. We provide [a util script](data/merge_data.py) to merge data from multiple splits to one split. An example to run the script is as follows:

```
python3 -m llmshearing.data.merge_data \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR \
        --output_split eval_merge \
        --split_names domain1 domain2
```

