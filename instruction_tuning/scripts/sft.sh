# REQUIRE: PORT
if [ -z "$SLURM_NTASKS_PER_NODE" ]
then
    SLURM_NTASKS_PER_NODE=$(expr $SLURM_NTASKS / $SLURM_NNODES)
fi

export WORLD_SIZE=$(expr $SLURM_NTASKS_PER_NODE \* $SLURM_NNODES)
FIRSTNODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$FIRSTNODE
export MASTER_PORT=$PORT
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$(expr $SLURM_NODEID \* $SLURM_NTASKS_PER_NODE + $SLURM_LOCALID)
export OMP_NUM_THREADS=10

echo master $MASTER_ADDR, port $MASTER_PORT, world size $WORLD_SIZE, local rank $LOCAL_RANK, rank $RANK

module purge
module load anaconda3/2022.10
conda activate ShareGPT

output_dir=$1
run_name=$2
model_name_or_path=$3
dataset_path=$4
epoch=$5

export WANDB_MODE=offline
python supervised.py \
  --model_name_or_path "${model_name_or_path}" \
  --eval_size 10 \
  --fp16 False \
  --bf16 True \
  --seed 42 \
  --output_dir "${output_dir}" \
  --dataset_path "${dataset_path}" \
  --num_train_epochs "${epoch}" \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --eval_steps 100 \
  --save_strategy "steps" \
  --save_steps 1000000000 \
  --save_total_limit 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --evaluation_strategy "steps" \
  --logging_steps 10 \
  --wandb_project "ShareGPT" \
  --run_name "${run_name}" \
  --tf32 True \
  --flash_attn True \
  --model_max_length 512 \
  --ddp_timeout 1800 \
  --fsdp "full_shard auto_wrap" \
  --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer"
