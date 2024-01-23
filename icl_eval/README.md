### How to run the evaluations

Please install the lm-eval package from the `lm-evaluation-harness` repository using the following commad:
```
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```
After installation, please refer to `run_eval.sh` to check out how to run evaluations with the pruned models and source models. 

### Metrics for each task
We use the following metrics for each task:

| Task              | Metric                |
|------------------|----------------------|
| arc_challenge    | acc_norm             |
| arc_easy         | acc                  |
| lambada_openai   | acc                  |
| logiqa           | acc_norm             |
| piqa             | acc                  |
| sciq             | acc                  |
| winogrande       | acc                  |
| wsc              | acc                  |
| hellaswag        | acc_norm |
| truthfulqa_mc    | mc2     |
| mmlu             | acc          |
| gsm8k            | acc          |
| boolq            | acc        |
| nq_open          | exact     |
