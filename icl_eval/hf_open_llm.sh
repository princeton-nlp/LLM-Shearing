#!/bin/bash
harness_dir=/scratch/gpfs/mengzhou/space2/llm_eval/scripts/harness

bsz="${bsz:-16}"
resultpath="${4:-tmp}"
echo $resultpath

hareness_output_dir=$n/space2/llm_eval/scripts/harness

cmd="python3 $harness_dir/main.py --model=hf-causal --model_args="pretrained=$1,dtype=float16" --tasks=$2 --num_fewshot=$3 --batch_size=$bsz --output_path=$harness_dir/result/$resultpath --no_cache"
if [[ -n $5 ]]; then cmd="$cmd --limit=$5"; fi

$cmd 