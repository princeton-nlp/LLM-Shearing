model=$1
modelname=$(basename "$model")

# sbatch --output=$harness_dir/slurm/%A-%x.out -N 1 -n 1 --mem=200G --cpus-per-task 10  --gres=gpu:a100:1 --mail-type=FAIL,TIME_LIMIT --mail-user=mengzhou@cs.princeton.edu --time 1:00:00 --job-name harnesspythia-$modelname -x "della-i14g[1-20]" <<EOF
# #!/bin/bash

# EOF

bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 0 pythia0shot-$modelname
bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 5 pythia5shot-$modelname