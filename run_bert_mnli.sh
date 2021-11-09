
#This example, show fine-tune BERT base on mnli use allennlp 

export CUDA_VISIBLE_DEVICES=7
allennlp train configs/nli/mnli_bert_base.jsonnet \
-s output_models/bert_base_mnli \
--include-package my_package

# This example, show evaluation BERT base on mnli mismatched
allennlp evaluate_mult output_models/bert_base_mnli/model.tar.gz \
 data/nli/multinli_1.0_dev_mismatched.jsonl \
 --output-file outputs_eval_mnli/multinli_result.txt  \
 --cuda-device 0 \
 --include-package my_package

# This example, show evaluation BERT base on heuristics
allennlp predict output_models/bert_base_mnli/model.tar.gz \
 data/nli/heuristics_evaluation_set.jsonl \
 --output-file outputs_eval_mnli/hans_result.jsonl \
 --batch-size 16 \
 --cuda-device 0 \
 --predictor textual_entailment \
 --include-package my_package

python utils/hans_parser.py \
 -i outputs_eval_mnli/hans_result.jsonl \
 -o outputs_eval_mnli/hans.out
 
python utils/evaluate_heur_output.py \
 outputs_eval_mnli/hans.out > outputs_eval_mnli/hans_results.txt