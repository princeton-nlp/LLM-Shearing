NUM_FILES=$(wc -l < jsonl_list.txt)
echo "Total files: $NUM_FILES"

for ((i=0; i<NUM_FILES; i++))
do
    echo $i
    SLURM_ARRAY_TASK_ID=$i python tokenize_single_file.py --target_dir tokenized_sample_redpajama --raw_dir sample_redpajama
done
