
#This example, show fine-tune BERT base on paraphrase_identification use allennlp 

export CUDA_VISIBLE_DEVICES=5
# allennlp train configs/paraphrase_identification/qqp_bert_base_1.jsonnet \
# -s output_models/bert_base_qqp \
# --include-package my_package

# This example, show evaluation BERT base on paraphrase_identification
allennlp evaluate output_models/bert_base_qqp/model.tar.gz \
 data/paraphrase_identification/paws.dev_and_test.jsonl \
 --output-file outputs_eval_qqp/paws_dev_and_test.json  \
 --cuda-device 0 \
 --include-package my_package

allennlp evaluate output_models/bert_base_qqp/model.tar.gz \
 data/paraphrase_identification/qqp.dev.jsonl \
 --output-file outputs_eval_qqp/qqp_dev.json  \
 --cuda-device 0 \
 --include-package my_package
