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

For HANS , we use evaluate_heur_output.py script  from [tommccoy1/hans](https://github.com/tommccoy1/hans) to evaluate a model's predictions. Here's an example with our counterfactual predictor:

```shell
allennlp cf_predict $MODEL_DIR/model.tar.gz data/nli/heuristics_evaluation_set_sample_weight.jsonl --output-file $MODEL_DIR/hans_predictions.jsonl --batch-size 64 --cuda-device 0 --cf_type counterfactual_snli --cf_weight  0.3745401188473625 --predictor cf_textual_entailment_reweighted --include-package my_package
cd utils/
python hans_parser.py -i $MODEL_DIR/hans_predictions.jsonl -o $MODEL_DIR/hans_predictions.out
python evaluate_heur_output.py $MODEL_DIR/hans_predictions.out > $MODEL_DIR/hans_result.txt
```

## In Details

1. Steps for running CMA [Can]<br/>
    a. How to get the files ready? <br/>
    b. How to train a bias model?  [Jab*, Korn*]  (which file to run, outputfile name)<br/>
    c. How to load a trained bias model (example) [Jab, Korn] <br/>
    d. How to train a main model  [Can*,Jab*, Korn*]  (which file to run, outputfile name)<br/>
    e. How to load a trained main model [Can, Jab]<br/>
    f. How to load model from a huggingface  [Korn] <br/>
2. Getting predictions:<br/>
    a. Get predictions from bias models [Jab,Korn] + jsonl files<br/>
        i. Jsonl train*<br/>
        ii. Jsonl dev<br/>
        iii. Jsonl test<br/>
        iv. Jsonl challenge set<br/>
    b. Get prediction from main models [Can, Jab] + jsonl files<br/>
        i. Slurm files for getting raw pred<br/>
        ii. Raw val set<br/>
        iii. Raw test set<br/>
        iv. Raw challenge set<br/>
4. Apply CMA [Can]<br/>
    a. Sharpness control (need predictions on valset for both models) [Can]<br/>
    b. TIE_A [Can]<br/>
    c. TE_model [Can]<br/>

## License
[MIT](https://choosealicense.com/licenses/mit/)

