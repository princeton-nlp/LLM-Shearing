model=$1
modelname=$(basename "$model")

# header for running slurm jobs for evaluation
# sbatch --output=$harness_dir/slurm/%A-%x.out -N 1 -n 1 --mem=200G --cpus-per-task 10  --gres=gpu:a100:1 --mail-type=FAIL,TIME_LIMIT --mail-user=mengzhou@cs.princeton.edu --time 1:00:00 --job-name harnesspythia-$modelname -x "della-i14g[1-20]" <<EOF
# #!/bin/bash
# EOF

# zero-shot evaluation for Pythia evaluation tasks
bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 0 pythia0shot-$modelname


# five-shot evaluation for Pythia evaluation tasks
bash hf_open_llm.sh $model lambada_openai,piqa,winogrande,wsc,arc_challenge,arc_easy,sciq,logiqa 5 pythia5shot-$modelname


# HF leaderboard evaluation
bash $harness_dir/hf_open_llm.sh $model hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions 5 mmlu5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model hellaswag 10 hellaswag5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model arc_challenge 25 arcc5shot-$modelname
bash $harness_dir/hf_open_llm.sh $model truthfulqa_mc 0 truthfulqa5shot-$modelname

# others
bash $harness_dir/hf_open_llm.sh $model nq_open 32 nq_open32shot-$modelname
bash $harness_dir/hf_open_llm.sh $model boolq 32 boolq32shot-$modelname
bash $harness_dir/hf_open_llm.sh $model gsm8k  8 gsm8k8shot-$modelname
