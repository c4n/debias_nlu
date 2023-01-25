# Mitigating Spurious Correlation in Natural Language Inference with Causal Inference

This GitHub repo contains codes and scripts for the paper "Mitigating Spurious Correlation in Natural Language Inference with Causal Inference" (EMNLP 2022).
### Counterfactual Inference Example
- The example in "counterfactual_inference_example_huggingface.ipynb" shows how one may use counterfactual inference to debias an existing NLI model.
- In order to get counterfactual inference results on MNLI as seen on paper "notebooks/Counterfactual_inference_debias_result_correction_Anon.ipynb" shows how one can apply counterfactual inference and collect results.
- To train bias models and main models from scratch, one may consult the rest of this readme file:



## Installation

- [allennlp 2.5.0](https://github.com/allenai/allennlp/tree/v2.5.0)



```shell
pip install allennlp==2.5.0 allennlp-models==2.5.0
```

## Usage

We mainly use CLI to run all the scripts.  We also use the slurm system; the slurm scripts are basically shell scripts with extra configurations. 

To include customized allennlp's packages add ``--include-package my_package'' to the command

### Training 

```shell
allennlp train <training_config>.jsonnet -s <output_path> --include-package my_package
```

### Evaluation

We can evaluate multiple evaluation datasets at once using ``'evaluate_mult''

```shell
MNLI_PARAMS=($MODEL_DIR/model.tar.gz  
<evaluation_set_a>.jsonl:<evaluation_set_b>.jsonl
--output-file=$MODEL_DIR/result_a.txt:$MODEL_DIR/result_b.txt
--cuda-device=0
--include-package=my_package)
allennlp evaluate_mult ${MNLI_PARAMS[@]}
```


## In Details


### Training and load a bias model

#### MNLI
- Create customized features for training the bias model. The example in "notebooks/Build_features_extraction.ipynb".
-  For the bias model used in the paper, see  "notebooks/Bias_Model_3_classes.ipynb" or "notebooks/Bias_Model_use_our_features.ipynb".

#### FEVER
Firstly, we need to make sure that the dataset is well placed in the relative path "data/fact_verification". For convenient, you can run the "download.sh" and "preprocess.sh" scripts in the path "data/fact_verification" to get a FEVER dataset. In order to train the bias model for FEVER dataset, you can configure the following parameters in "notebooks/Bias_Model_FEVER.ipynb" file. Then we run all the python script in this file for training the bias model and save it into your pointed path.

```bash
DUMMY_PREFIX = "" # "sample_" for few samples and "" for the real one

TRAIN_DATA_FILE = "../data/fact_verification/%sfever.train.jsonl"%DUMMY_PREFIX
VAL_DATA_FILE = "../data/fact_verification/%sfever.val.jsonl"%DUMMY_PREFIX
DEV_DATA_FILE = "../data/fact_verification/%sfever.dev.jsonl"%DUMMY_PREFIX
TEST_DATA_FILE = "../data/fact_verification/fever_symmetric_v0.1.test.jsonl"
```

```bash
WEIGHT_KEY = "sample_weight"
OUTPUT_VAL_DATA_FILE = "../data/fact_verification/%sweighted_fever.val.jsonl"%DUMMY_PREFIX
OUTPUT_TRAIN_DATA_FILE = "../data/fact_verification/%sweighted_fever.train.jsonl"%DUMMY_PREFIX
SAVED_MODEL_PATH = "../results/fever/bias_model"
```

In addition, the example process of loading bias model is also contains in "notebooks/Bias_Model_FEVER.ipynb".

#### QQP
- Create features for training the bias model. The example in "notebooks/qqp_features_extraction.ipynb".
- To train the bias model used in the paper, see: "notebooks/qqp_feature_classification_using_MaxEnt.ipynb".


## How to train a main model

Once you have the scripts and the dataset available on your local machine, the training process of the main model could be execute via the following running template.

```bash
bash slurm_jobs/{DATASET}/job_{MODEL_NAME}_train.sub
```

For example, training the baseline model with FEVER dataset would requires you to execute this command.

```bash
bash slurm_jobs/fact_verification/job_bert_base_train.sub
```


## How to evaluate a main model

Simialar to training, to evaluate the trained model on the testset, we could do it by executing the following command.

```bash
bash slurm_jobs/{DATASET}/job_{MODEL_NAME}_eval.sub
```

## How to get raw prediction

Getting raw prediction from trained model on FEVER is quite trivial when using the "job_get_raw.sub" and following by your path of the trained model. For instance,

```bash
bash slurm_jobs/fact_verification/job_get_raw.sub results/outputs_fever_bert_base_1
```

- for raw prediction data: you can download using the following link: [https://anonymshare.com/2QL1/pred-data.zip](https://anonymshare.com/2QL1/pred-data.zip)


## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgement
We thank [Sitiporn Sae-Lim](https://github.com/sitiporn) for refactoring our code. 
