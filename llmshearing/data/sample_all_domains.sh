NUM_DOMAINS=7
echo "Total domains: $NUM_DOMAINS"

for ((i=0; i<NUM_DOMAINS; i++))
do
    echo $i
    SLURM_ARRAY_TASK_ID=$i python sample.py --target_dir mds_sample_redpajama --tokenized_dir tokenized_sample_redpajama
done
