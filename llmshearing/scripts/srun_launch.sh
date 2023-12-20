PROJ_DIR=$n/space2/LLM-Shearing

echo ${SLURM_NODEID} 
composer --node_rank ${SLURM_NODEID} $PROJ_DIR/llmshearing/train.py $@  