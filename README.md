# Mitigating Spurious Correlation in Natural Language Inference using Causal Inference

This GitHub repo contains codes and scripts for the paper xxxx.

## Installation

- [allennlp 2.5.0](https://github.com/allenai/allennlp/tree/v2.5.0)

to do...

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

### Steps for running CMA [Can]
### How to get the files ready?

### Training and load bias model
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


#### How to train a main model  [Can*,Jab*, Korn*]  (which file to run, outputfile name)
#### How to load a trained main model [Can, Jab]
#### How to load model from a huggingface  [Korn]
        
### Getting predictions:
#### Get predictions from bias models [Jab,Korn] + jsonl files
##### Jsonl train*
##### Jsonl dev
##### Jsonl test
##### Jsonl challenge set
#### Get prediction from main models [Can, Jab] + jsonl files
##### Slurm files for getting raw pred
##### Raw val set
##### Raw test set
##### Raw challenge set
### Apply CMA [Can]
#### Sharpness control (need predictions on valset for both models) [Can]
#### TIE_A [Can]
#### TE_model [Can]


## License
[MIT](https://choosealicense.com/licenses/mit/)

