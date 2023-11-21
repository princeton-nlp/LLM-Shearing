# An example for 2 nodes, each uses 4 GPUs
run_name="sft10K"
epoch=2
output_dir="../../ShareGPT512results/LLaMA2_test" # save_dir
dataset_path="../../ShareGPT512" # `sft.json` and `test.json` are in this folder.
model_name_or_path=princeton-nlp/Sheared-LLaMA-1.3B

sbatch --output=slurm/%A_%a-%x.out \
    -N 2 \
    --ntasks-per-node 4 \
    --mem=128G \
    --cpus-per-task 10 \
    --gres=gpu:a100:4 \
    --constraint gpu80 \
    --time 0:59:59 \
    --array 0-0 \
    --job-name sft -x "della-i14g[1-20]" <<EOF
#!/bin/bash
PORT=\$(expr \$RANDOM + 1000) srun --wait 0 bash scripts/sft.sh $output_dir $run_name $model_name_or_path $dataset_path $epoch
EOF
