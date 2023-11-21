decoder_name_or_path="../../ShareGPT512results/LLaMA2_test" # save_dir
dataset_path="../../ShareGPT512" # `sft.json` and `test.json` are in this folder.
dataset_name="test"
output_path_to_store_samples="${dataset_name}/temp0.0_num1.json"
num_return_sequences="1"
temperature="0.0"

sbatch --output=slurm/%A_%a-%x.out \
    -N 1 \
    --ntasks-per-node 1 \
    --mem=128G \
    --cpus-per-task 10 \
    --gres=gpu:a100:1 \
    --constraint gpu80 \
    --time 0:59:59 \
    --array 0-0 \
    --job-name inference -x "della-i14g[1-20]"  <<EOF
#!/bin/bash
srun --wait 0 bash scripts/inference.sh $decoder_name_or_path $dataset_path $dataset_name $output_path_to_store_samples $num_return_sequences $temperature
EOF
