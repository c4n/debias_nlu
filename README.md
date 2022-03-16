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
#### How to get the files ready?
#### How to train a bias model?  [Jab*, Korn*]  (which file to run, outputfile name)
#### How to load a trained bias model (example) [Jab, Korn]
#### How to train a main model  [Can*,Jab*, Korn*]  (which file to run, outputfile name)
#### How to load a trained main model [Can, Jab]
#### How to load model from a huggingface  [Korn]
        
### Getting predictions:<br/>
### Get predictions from bias models [Jab,Korn] + jsonl files
#### Jsonl train*
#### Jsonl dev
#### Jsonl test
#### Jsonl challenge set
### Get prediction from main models [Can, Jab] + jsonl files
#### Slurm files for getting raw pred
#### Raw val set
#### Raw test set
#### Raw challenge set
### Apply CMA [Can]
#### Sharpness control (need predictions on valset for both models) [Can]
#### TIE_A [Can]
#### TE_model [Can]

## License
[MIT](https://choosealicense.com/licenses/mit/)

