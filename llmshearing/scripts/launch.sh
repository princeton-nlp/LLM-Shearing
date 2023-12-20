#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=gputest
#SBATCH --time=1:00:00

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=512gb
#SBATCH --constraint gpu80
#SBATCH --output=/scratch/gpfs/mengzhou/space2/out/logs/%x-%j.out

PROJ_DIR=$n/space2/LLM-Shearing
LOG_DIR=/scratch/gpfs/mengzhou/space2/out/logs

# num_nodes=$(scontrol show job $SLURM_JOB_ID | grep NodeList=della | wc -l)
num_nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

export MASTER_ADDR=$master_addr
echo $SLURM_GPUS_PER_NODE

export WORLD_SIZE=$(( $num_nodes * $SLURM_GPUS_PER_NODE ))
export MASTER_PORT=$(( 10000 + RANDOM % 10000 ))

echo "MASTER_ADDR="$MASTER_ADDR
echo "MASTER_PORT="$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE
echo "num_nodes="$num_nodes

if [[ $num_nodes == 1 ]]; then composer $PROJ_DIR/llmshearing/train.py $@; 
else srun --output=$LOG_DIR/%x-%j-%n.out bash $PROJ_DIR/llmshearing/scripts/srun_launch.sh $@; fi
 
