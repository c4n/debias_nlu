#!/bin/bash -l

# DATASET=${1:-paraphrase_identification}
DATASET=${1:-fact_verification}

# MODEL_PATHS=(
#     "results/outputs_qqp_bert_base_1"
#     "results/outputs_qqp_weighted_bert_1"
#     "results/outputs_qqp_poe_bert_1"
# )
# MODEL_PATHS=(
#     "results/fever/baseline_lr2e5/outputs_fever_bert_base_1"
#     "results/fever/reweight_lr2e5/outputs_fever_weighted_bert_1"
#     "results/fever/poe_lr2e5/outputs_fever_poe_bert_1"
# )
MODEL_PATHS=(
    "results/fever/self_distill_lr2e5/outputs_fever_self_distill_bert_1"
)

SEEDs=( 2 3 4 5 6 )

for model_path in "${MODEL_PATHS[@]}"
do
    for seed in "${SEEDs[@]}"
    do
        bash slurm_jobs/${DATASET}/job_get_raw.sub ${model_path}_seed${seed}3370
    done
done

