#!/bin/bash -l


## Train
# bash slurm_jobs/fact_verification/job_bert_base_train.sub
# bash slurm_jobs/fact_verification/job_poe_bert_train.sub
# bash slurm_jobs/fact_verification/job_reweighting_bert_train.sub
# bash slurm_jobs/fact_verification/job_self_distill_utama_base_train.sub

# bash slurm_jobs/paraphrase_identification/job_bert_base_train.sub
# bash slurm_jobs/paraphrase_identification/job_poe_bert_train.sub
# bash slurm_jobs/paraphrase_identification/job_reweighting_bert_train.sub
bash slurm_jobs/paraphrase_identification/job_self_distill_utama_base_train.sub

## Eval
# bash slurm_jobs/fact_verification/job_bert_base_eval.sub
# bash slurm_jobs/fact_verification/job_poe_bert_eval.sub
# bash slurm_jobs/fact_verification/job_reweighting_bert_eval.sub
# bash slurm_jobs/fact_verification/job_self_distill_utama_base_eval.sub

# bash slurm_jobs/paraphrase_identification/job_bert_base_eval.sub
# bash slurm_jobs/paraphrase_identification/job_poe_bert_eval.sub
# bash slurm_jobs/paraphrase_identification/job_reweighting_bert_eval.sub
bash slurm_jobs/paraphrase_identification/job_self_distill_utama_base_eval.sub
