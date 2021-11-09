
#This example, show fine-tune BERT base on fact_verification use allennlp 

export CUDA_VISIBLE_DEVICES=7
allennlp train configs/fact_verification/fever_bert_base_1.jsonnet \
-s output_models/bert_base_fever \
--include-package my_package

# This example, show evaluation BERT base on fever_symmetric_generated
allennlp evaluate output_models/bert_base_fever/model.tar.gz \
 data/fact_verification/fever_symmetric_generated.jsonl \
 --output-file outputs_eval_fever/fever_symmetric_generated.json  \
 --cuda-device 0 \
 --include-package my_package

# This example, show evaluation BERT base on fever
allennlp evaluate output_models/bert_base_fever/model.tar.gz \
 data/fact_verification/fever.dev.jsonl \
 --output-file outputs_eval_fever/fever_dev.json  \
 --cuda-device 0 \
 --include-package my_package