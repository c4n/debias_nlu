import pandas as pd 
import jsonlines
import sys, getopt
import numpy as np
from scipy.special import softmax

from ray import tune
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.bayesopt import BayesOptSearch
from sklearn.metrics import log_loss


mnli_target_dict = {'entailment':0,'neutral':2,'contradiction':1}


def counterfactual_function(config):
    e_sample = 0
    e_correct = 0
    ne_sample = 0
    ne_correct = 0 
    
    cf_weight = config["cf_weight"]
    entropy_curve = config["entropy_curve"]
    for line, line_gt in zip(f1,f2):

        logit, logit_cf= line['logits'], line['cf_logits']
        gt, sample_weight = line_gt['gold_label'], line_gt['sample_weight']
            
            
        # entropy switch
        x=softmax(logit)
        entropy = -sum(x*np.log(x)/np.log(3))
#   * 0.2
        # end switch    
        out_logit = np.array(logit)-((entropy**entropy_curve)*cf_weight*np.array(logit_cf))
        if gt == "-":
            continue
#             out_logit = np.array(logit)- cf_weight*np.array(logit_cf)
        if sample_weight >= 0.81 and mnli_target_dict[gt] == 0:
#                 e += 1
            if np.argmax(softmax(out_logit)) == gt:
                e_correct += 1 
            e_sample +=1  
        elif sample_weight >= 0.75 and mnli_target_dict[gt] != 0:
#                 ne += 1
            if np.argmax(softmax(out_logit)) != 0:
                ne_correct += 1 
            ne_sample +=1   

    total_score = ((ne_correct+e_correct)/(ne_sample+e_sample))   
 
    tune.report(score = total_score) 

def main(argv):
    output_path = ''

    
    try:
       opts, args = getopt.getopt(argv,"hm:d:o:",["help","model_dev_result=","dev_file=","output_path="])
    except getopt.GetoptError:
       print('mnli_ray_tune.py -m <model_dev_result> -d <dev_file> -o <output_path>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('mnli_ray_tune.py -m <model_dev_result> -d <dev_file> -o <output_path>')
          sys.exit()
       elif opt in ("-m", "--model_dev_result"):
          model_dev_result = arg    
    #    elif opt in ("-d", "--dev_file"):
        #   pred_path = arg               
       elif opt in ("-o", "--output_path"):
          output_path = arg



    # load files here
    global f1
    global f2
    f1= jsonlines.open(model_dev_result)
    f2_temp = jsonlines.open('/ist/users/canu/debias_nlu/data/nli/multinli_1.0_dev_matched_overlap.jsonl') 
    f1 = [line for line in f1.iter()]
    f2 = []
    f2_temp = [gt for gt in f2_temp.iter()]
    for gt in f2_temp:
        f2.append({'gold_label':gt['gold_label'],'sample_weight':gt['sample_weight'] })


    config = {"cf_weight": tune.uniform(0, 8),
                "entropy_curve": tune.uniform(0, 8)}

    analysis = tune.run(counterfactual_function,
            config=config, 
            metric="score", 
            mode="max",
        # Limit to two concurrent trials (otherwise we end up with random search)
        search_alg=ConcurrencyLimiter(
            BayesOptSearch(random_search_steps=2),
            max_concurrent=2),
        num_samples=20,
        stop={"training_iteration": 30},
        verbose=2)

    print("Best config: ", analysis.get_best_config(
        metric="score", mode="max"))

    output = analysis.get_best_config(metric="score", mode="max")


    with open(output_path+'/cf_weight.txt', 'w') as f:
        f.write(str(output['cf_weight']))

    with open(output_path+'/entropy_curve.txt', 'w') as f:
        f.write(str(output['entropy_curve']))  




if __name__ == "__main__":
    main(sys.argv[1:])
          