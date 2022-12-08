import os
from typing import Callable, List, Union, Tuple, Dict
import fuse, causal_utils
import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.special import expit
from sklearn.metrics import f1_score
from ASD import ASD
import fuse
import glob
from kl_general import sharpness_correction, TE_CONFIG as ESTIMATE_C_TE_CONFIG , DEFAULT_CONFIG as ESTIMATE_C_DEFAULT_CONFIG
import pickle


PROB_T = Union[List[float], List[List[float]]]

BIAS_MODEL_DICT = {
    "mnli_train": "train_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    "mnli_dev_mm": "test_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    "mnli_hans": "hans_prob_korn_lr_overlapping_sample_weight_3class.jsonl",

    "fever_train": "weighted_fever.train.jsonl",

    "fever_dev": "weighted_fever.dev.jsonl",
    "fever_sym1": "weighted_fever_symmetric_v0.1.test.jsonl",
    "fever_sym2": "weighted_fever_symmetric_v0.2.test.jsonl",
    "qqp_train": "qqp_train_overlap_only_bias_weighted.jsonl",

    "qqp_dev": "qqp_dev_overlap_only_bias_weighted.jsonl", 
    "qqp_paws": "paws_dev_and_test_overlap_only_bias_weighted.jsonl", 
}


TASK2TRAIN_DICT = {"nli": "mnli_val", "fever": "fever_val", "qqp": "qqp_val"}

BERT_MODEL_RESULT_DICT = {

    "mnli_train": "raw_train.jsonl",
    "mnli_val": "raw_m.jsonl",

    "mnli_dev_mm": "raw_mm.jsonl",
    "mnli_hans": "normal/hans_result.jsonl",
    
    "fever_train": "raw_fever.train.jsonl",
    "fever_val": "raw_fever.val.jsonl",

    "fever_dev": "raw_fever.dev.jsonl",
    "fever_sym1": "raw_fever_symmetric_v0.1.test.jsonl",
    "fever_sym2": "raw_fever_symmetric_v0.2.test.jsonl",

    "qqp_train": "raw_qqp.train.jsonl",
    "qqp_val": "raw_qqp.val.jsonl",

    "qqp_dev": "raw_qqp.dev.jsonl",  
    "qqp_paws": "raw_paws.dev_and_test.jsonl", 
}

class Inference:

    def __init__(self, data_path: str,
    model_path: str,
    task: str,
    test_set: str = None,
    MODE_PATH_CONFIG: dict = None,
    TE_CONFIG: dict = None,
    DEFAULT_CONFIG: dict = None,
    label_maps: dict = None,
    DEBUG: bool = False,
    ground_truth_key: str = "gold_label",
    fusion: Callable[[PROB_T], PROB_T] = None,
    bias_val_pred_file: str = "dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    model_val_pred_file: str = "raw_m.jsonl")-> None:

        # Todo: add explaination' arguments of this class 
        self.data_path  = data_path
        self.model_path = model_path
        self.task = task
        self.mode_path_config = MODE_PATH_CONFIG
        self.te_config = TE_CONFIG
        self.default_config = DEFAULT_CONFIG
        self.fusion = fusion
        self.label_maps = label_maps
        self.ground_truth_key = ground_truth_key
        self.DEBUG = DEBUG

        self.bias_val_pred_file = bias_val_pred_file

        if DEBUG:
            print(f"self.bias_val_pred_file : {self.bias_val_pred_file}")
        
        
        self.model_val_pred_file = model_val_pred_file

        self.df_bert = {}
        self.df_bias = {} 

        self.y_prob = { 'a' : {'x' :  None, 'x*':  None}}

        self.label_modes = {"normal": None, "TIE_debias": None,"TE_debias": None}

        self.modes = list(self.mode_path_config.keys())
        
        for mode in self.modes:

            self.bert_path = os.path.join(
                        self.data_path, 
                        self.task, 
                        self.model_path, 
                        self.mode_path_config[mode][0])
                        
            self.bias_path =  os.path.join(
                        self.data_path, 
                        self.task,
                        self.mode_path_config[mode][1]) 
    
            if DEBUG:
                print(f"bert path : {self.bert_path}")
                print(f"bias path : {self.bias_path}")

            self.df_bert[mode] = pd.read_json(self.bert_path ,lines=True)
            
            self.df_bias[mode] = pd.read_json(self.bias_path ,lines=True)               

            if mode == "train":

                self.bias_val_pred_file  = self.mode_path_config["train"][1] 
                self.model_val_pred_file = self.mode_path_config["train"][0] 
            
            elif mode == "eval": 

                self.bert_probs = self.df_bert[mode]['probs']                         
                self.bias_probs = np.array([b for b in self.df_bias[mode]['bias_probs']])
    
    def get_probs(self)-> Tuple:

        return self.bert_probs, self.bias_probs
             
    def get_tie_scores(self)-> List:
                                
        
        self.y_prob['a']['x']  = []


        self.default_config["N_LABELS"] = self.bias_probs.shape[-1]

        if self.DEBUG:
            print(f"in get tie_scores")
            print(f"self.bias_val_pred_file : {self.bias_val_pred_file}")
            print(f"self.model_val_pred_file: {self.model_val_pred_file}")


        TIE_A = []
        
        c = get_c(
                data_path  = os.path.join(self.data_path, self.task),
                model_path = os.path.join(self.data_path, self.task, self.model_path), 
                fusion = self.fusion,
                DEBUG = self.DEBUG,
                bias_val_pred_file  = self.bias_val_pred_file,
                model_val_pred_file = self.model_val_pred_file,
                config = self.default_config)
                                            
        for p, h in zip(self.bert_probs, self.bias_probs):
            
            new_y1m1 = self.fusion(np.array(p), h)

            self.y_prob['a']['x'].append(new_y1m1)

        self.y_prob['a']['x*'] = self.fusion(c, self.bias_probs)

        for p, b in zip(self. y_prob['a']['x'], self. y_prob['a']['x*']):

            TIE_A.append(p - b)
            
        return TIE_A, c

    def get_te_model(self)-> List:
        
        """
        TE_m : p <- bert_probs,  for TE there is no  fuse
               b <- bias_probs,
             : p - c2 * b 
        """     
        
        TE_model = []


        
        self.te_config["N_LABELS"] = self.bias_probs.shape[-1]
        
        # this c is no fusion
        c = get_c(os.path.join(self.data_path, self.task), 
                  os.path.join(self.data_path, self.task, self.model_path), 
                  DEBUG = self.DEBUG,
                  bias_val_pred_file  = self.bias_val_pred_file,
                  model_val_pred_file = self.model_val_pred_file,
                  config = self.te_config)  
        
        for p, b in zip(self.bert_probs, self.bias_probs):    
         
            # TE_model.append(p - b)
            TE_model.append(p - c * b)
        
        return TE_model, c
                    
    def get_text_answers(self)-> Dict:
        
        self.label_modes['normal'] = self.bert_probs
        self.label_modes['TIE_debias'], tie_c = self.get_tie_scores()
        self.label_modes['TE_debias'], te_c  = self.get_te_model()
        
        text_answers = {}
        
        for label_mode in (list(self.label_modes.keys())):

            labels = []

            for cur_distribution in self.label_modes[label_mode]:
                labels.append(self.label_maps[np.argmax(cur_distribution)])

            self.df_bert[label_mode] = labels
            text_ans = ""
            
            for idx, obj in enumerate(self.df_bert[label_mode]):
                text_ans = text_ans + "ex"+str(idx)+","+obj+"\n"

            text_answers[label_mode] = text_ans
            
        return text_answers   

    def get_unique_label(self) -> Tuple:

        if self.DEBUG:
            print(f"df_bias keys : {self.df_bias.keys()}")
            print(f"df_bias keys : {self.df_bias['eval'].keys()}")
            print(f"unique labels : {self.df_bias['eval']['gold_label'].unique()}")

        labels = self.df_bias['eval'][self.ground_truth_key]
        unique_labels = labels.unique().tolist()

        offset = 0
        
        
        if "-" in self.df_bias['eval'][self.ground_truth_key].value_counts():
            # no ground truth
            offset = self.df_bias['eval'][self.ground_truth_key].value_counts()["-"]
        
        if self.DEBUG:
            print("unique_labels: ", unique_labels)

        return unique_labels, labels, offset
                                 
    def format_label(self, label)-> str:
        
        if label == "entailment":
            return "entailment"
        else:
            return "non-entailment"

    def get_guess_dict(self)-> Dict:

        all_guess_dict = {}
        text_answers = self.get_text_answers()
        
        for mode in (list(text_answers.keys())):
            guess_dict = {}
            for line in text_answers[mode].split("\n"):
                if len(line)>1:
                    parts = line.strip().split(",")
                    guess_dict[parts[0]] = self.format_label(parts[1])

            all_guess_dict[mode] = guess_dict

        return all_guess_dict    

def get_ans(ans: int, test_set: str) -> Union[int, str]:
    if test_set == "mnli_hans":
        if ans == 0:
            return "entailment"
        else:
            return "non-entailment"

    if test_set == "mnli_test" or test_set == "mnli_dev_mm" or test_set == "mnli_dev_m":
        gt_key = {0: "entailment", 1: "contradiction", 2: "neutral"}
        return gt_key[ans]

    if "fever" in test_set:
        gt_key = {0: "SUPPORTS", 1: "NOT ENOUGH INFO", 2: "REFUTES"}
        return gt_key[ans]

    if "qqp" in test_set:
        return ans

    raise NotImplementedError("Does not support test_set: %s" % test_set)


def get_bias_effect(
    nde: List[List[float]],
    nie: List[List[float]],
    tie: List[float],
    te: List[List[float]],
    test_set: str,
) -> Tuple[Union[List[float], List[List[float]]]]:
    if "nli" in test_set:
        return (nde[0], nie[0], tie[0], te[0])
    elif "fever" in test_set:
        return (nde[2], nie[2], tie[2], te[2])
    elif "qqp" in test_set:
        return (nde[1], nie[1], tie[1], te[1])
    raise NotImplementedError("Does not support test_set: %s" % test_set)

def get_c(
    data_path: str,
    model_path: str,
    x0: PROB_T = None,
    fusion: Callable[[PROB_T], PROB_T] = None,
    DEBUG: bool = False,
    bias_val_pred_file: str = "dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    model_val_pred_file: str = "raw_m.jsonl",
    bias_probs_key: str = "bias_probs",
    model_probs_key: str = "probs",
    config: dict = ESTIMATE_C_DEFAULT_CONFIG,
) -> List[float]:

    if DEBUG:
        print(f"In get_c function:")
        print(f"bias_val_pred_file:")
        print(os.path.join(data_path, bias_val_pred_file))
        print(f"model_val_pred_file")
        print(os.path.join(model_path, model_val_pred_file))
        print(f"current config : {config}")

    df_bias_dev = pd.read_json(os.path.join(data_path, bias_val_pred_file), lines=True)
    bias_dev_score = [b for b in df_bias_dev[bias_probs_key]]
    bias_dev_score = np.array(bias_dev_score)
    # ya1x0_dev = fusion(bias_dev_score, x0)

    df_bert_dev = pd.read_json(
        os.path.join(model_path, model_val_pred_file), lines=True
    )

    ya1x1prob_dev = []
    if fusion is not None:
        for p, h in zip(df_bert_dev[model_probs_key], bias_dev_score):
            new_ya1x1 = fusion(np.array(p), h)
            ya1x1prob_dev.append(new_ya1x1)
    

    if DEBUG:
        print(f"bias_dev_score : {bias_dev_score.shape}")
        print(f"bert_dev_score  : {df_bert_dev['probs'].shape}")
        # print(f"bias_dev_score val : {bias_dev_score[:3,:]}")

    c = sharpness_correction(bias_dev_score, df_bert_dev['probs'] if fusion is None else ya1x1prob_dev, config=config)
    
    n_labels = bias_dev_score[0].shape[0]
    c = c * np.ones(n_labels)

    if DEBUG:
        print("c: ", c)
        print("softmax(c): ", softmax(c))

    return c

def get_c_te(
    data_path: str,
    model_path: str,
    fusion: Callable[[PROB_T], PROB_T],
    x0: PROB_T,
    bias_val_pred_file: str = "dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    model_val_pred_file: str = "raw_m.jsonl",
    bias_probs_key: str = "bias_probs",
    model_probs_key: str = "probs",
    config: dict = ESTIMATE_C_TE_CONFIG,
) -> List[float]:
    print(os.path.join(
        data_path, bias_val_pred_file))
    df_bias_dev = pd.read_json(os.path.join(
        data_path, bias_val_pred_file), lines=True)
    bias_dev_score = [b for b in df_bias_dev[bias_probs_key]]
    bias_dev_score = np.array(bias_dev_score)
    # ya1x0_dev = fusion(bias_dev_score, x0)
    df_bert_dev = pd.read_json(
        os.path.join(model_path, model_val_pred_file), lines=True
    )
    
    c = sharpness_correction(bias_dev_score, df_bert_dev[model_probs_key], config=config)
    n_labels = bias_dev_score[0].shape[0]
    c = c*np.ones(n_labels)
    print("te_c: ", c)
    return c

def _default_model_pred(
    _input: List[float] = [[0, 0, 0.41997876976119086]],
    _model_name: str = "mnli_lr_model.sav",
) -> List[float]:
    loaded_model = pickle.load(open(_model_name, "rb"))
    return loaded_model.predict_proba(_input)


def report_CMA(
    model_path: str,
    task: str,  # MNLI, FEVER, QQP
    data_path: str,
    test_set: str,
    fusion: Callable[[PROB_T], PROB_T] = fuse.sum_fuse,
    input_a0: List[float] = [[0, 0, 0.41997876976119086]],
    estimate_c_config: dict = ESTIMATE_C_DEFAULT_CONFIG,
    estimate_c_te_config: dict = ESTIMATE_C_TE_CONFIG,
    correction: bool = False,
    DEBUG: bool = False,
    bias_probs_key: str = "bias_probs",
    ground_truth_key: str = "gold_label",
    model_pred_method: Callable[[List[float]], List[float]] = _default_model_pred,
    bias_val_pred_file: str = "dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    model_val_pred_file: str = "raw_m.jsonl",
    seed_path: List[str] = None,
    return_raw = False
) -> None:
    """
    Arguments:
        - test_set: test, challenge set
        - bias_val_pred_file: val set from bias model (including probs)
        - model_val_pred_file: val set from BERT model (including probs)
    """
    # load predictions from bias model (e.g., logistic regression)
    assert test_set in BIAS_MODEL_DICT.keys()



    # get a list of all seed dir
    if not seed_path:

        path_w_o_task = task if task == 'nli' else ""
        to_glob = data_path + f'{task}/' + model_path + path_w_o_task + "/*/"
        #data_path + f'{task}/' + model_path + task + "/*/"
        seed_path = glob.glob(to_glob)  # list of model dir for all seeds

    # init list to store results
    TE_explain = []
    TIE_explain = []
    factual_scores = []
    factual_f1 = {}
    TIE_scores = []
    TIE_f1 = {}
    NIE_explain = []
    NIE_scores = []
    NIE_f1 = {}
    INTmed_explain = []
    INTmed_scores = []
    INTmed_f1 = {}
    my_causal_query = []
    my_causal_f1 = {}

    # store raw pred/ TIE
    raw_factual_correct = []
    raw_TIE = []

    # get avg score
    for seed_idx in range(len(seed_path)):

        if DEBUG:
            print(
                os.path.join(
                    seed_path[seed_idx], BERT_MODEL_RESULT_DICT[TASK2TRAIN_DICT[task]]
                )
            )


        # MODE_PATH_CONFIG = {"eval": [ BERT_MODEL_RESULT_DICT["mnli_dev_mm"], BIAS_MODEL_DICT[test_set] ]}
        MODE_PATH_CONFIG = {"eval": [ BERT_MODEL_RESULT_DICT[test_set], BIAS_MODEL_DICT[test_set] ]}

        cur_seed_model_path =  seed_path[seed_idx].replace(data_path + f'{task}/',"")

        correction = True

        counterfactual = Inference( 
                            data_path,
                            model_path = cur_seed_model_path,
                            task = task, 
                            test_set = test_set,
                            MODE_PATH_CONFIG = MODE_PATH_CONFIG,
                            TE_CONFIG = estimate_c_te_config, 
                            DEFAULT_CONFIG = estimate_c_config,
                            fusion = fusion,
                            bias_val_pred_file = bias_val_pred_file,
                            model_val_pred_file = model_val_pred_file,
                            DEBUG = DEBUG)
        
        if DEBUG:
            print(f"current seed idx : {seed_idx}")


        debias_scores, tie_c = counterfactual.get_tie_scores()
        te_scores, te_c = counterfactual.get_te_model()

        if DEBUG:
            print(f"compute TIE : {np.array(debias_scores).shape} ")
            print(f"compute TE : {np.array(te_scores).shape}")
          
        unique_labels, labels, offset = counterfactual.get_unique_label()
        bert_probs, bias_probs = counterfactual.get_probs()


        a = {"treatment": None,"no-treatment": None}
        x = {"treatment": None,"no-treatment": None}
        
        y_prob = { 'a' : {'x' :  None, 'x*':  None},
                   'a*': {'x':   None, 'x*':  None}}

        n_labels = bias_probs.shape[1]

        # Todo: change accodingly if correction x0 <- get_c  
        x['no-treatment'] = (1/n_labels) * np.ones(n_labels)
        a['no-treatment'] = (1/n_labels) * np.ones(n_labels)

        te_correction = (1/n_labels) * np.ones(n_labels)

        if correction:
            x['no-treatment'] = tie_c #np.array([0.10714141, 0.10714141, 0.10714141])
            te_correction = te_c
        
        a['treatment'] = bias_probs
        x['treatment'] = bert_probs

        # fusion to create ya1x0
        # ya1x0prob = fusion(a1, x0)
        y_prob['a']['x*'] = fusion(a['treatment'], x['no-treatment'])
        y_prob['a*']['x*'] =  fusion(a['no-treatment'],x['no-treatment'])

        #ya0x0 = fusion(a0, x0)
        y_prob['a*']['x*'] = fusion(a['no-treatment'], x['no-treatment'])
        
        #ya1x1prob
        y_prob['a']['x'] = []

        for b, p in zip(a['treatment'], x['treatment']):
            cur_y_prob = fusion(np.array(b), p)
            y_prob['a']['x'].append(cur_y_prob)

        # to measure accuracy
        factual_pred_correct = []
        TIE_pred_correct = []
        NIE_pred_correct = []
        INTmed_pred_correct = []
        pred_correct = []

        # for mediation analysis
        all_TE = []
        all_TIE = []
        all_NIE = []
        all_NDE = []
        all_INTmed = []
        factual_y_preds = []
        tie_y_preds = []
        nie_y_preds = []
        intmed_y_preds = []
        my_causal_y_preds = []

        # Todo: vectorize
        for idx in range(len(labels)):

            # ya0x1 
            y_prob['a*']['x'] = fusion(a['no-treatment'], np.array(x['treatment'][idx]))

            # ya1x1 - ya0x0  
            TE  = y_prob['a']['x'][idx] - y_prob['a*']['x*']

            # ya1x1 - ya1x0 
            TIE = debias_scores[idx] 
            
            #ya1x0 - ya0x0
            NDE = y_prob['a']['x*'][idx] - y_prob['a*']['x*'] 
            
            #ya0x1 - ya0x0
            NIE = y_prob['a*']['x'] - y_prob['a*']['x*'] 
            
            INTmed = TIE - NIE
                
            # factual ## Todo: wrong index for FEVER
            factual_ans = np.argmax(x['treatment'][idx])

            

            factual_ans = get_ans(factual_ans, test_set)
            assert type(factual_ans) == type(labels[idx])

            factual_y_preds.append(factual_ans)
            factual_correct = factual_ans == labels[idx]
            # collect factual answer
            factual_pred_correct.append(factual_correct)

            # TIE
            TIE_ans = np.argmax(TIE) 
            TIE_ans = get_ans(TIE_ans, test_set)
            assert type(TIE_ans) == type(labels[idx])
            tie_y_preds.append(TIE_ans)
            TIE_correct = TIE_ans == labels[idx]
            TIE_pred_correct.append(TIE_correct)

            # INTmed
            INTmed_ans = np.argmax(INTmed)
            INTmed_ans = get_ans(INTmed_ans, test_set)
            assert type(INTmed_ans) == type(labels[idx])
            intmed_y_preds.append(INTmed_ans)
            INTmed_correct = INTmed_ans == labels[idx]
            INTmed_pred_correct.append(INTmed_correct)

            # NIE
            NIE_ans = np.argmax(NIE)
            NIE_ans = get_ans(NIE_ans, test_set)
            assert type(NIE_ans) == type(labels[idx])
            nie_y_preds.append(NIE_ans)
            NIE_correct = NIE_ans == labels[idx]
            NIE_pred_correct.append(NIE_correct)

            # save ## Todo: map bias class to correct index for each dataset
            bias_nde, bias_nie, bias_tie, bias_te = get_bias_effect(
                nde=NDE, nie=NIE, tie=TIE, te=TE, test_set=test_set
            )

            all_NDE.append(bias_nde)
            all_NIE.append(bias_nie)
            all_TIE.append(bias_tie)
            all_TE.append(bias_te)
            all_INTmed.append((INTmed[0]))

            entropy = -sum(
                bert_probs[idx] *
                np.log(bert_probs[idx]) / np.log(n_labels)
            )

            # TE_model
            cf_ans = np.argmax(np.array(te_scores[idx]))
            cf_ans = get_ans(cf_ans, test_set)
            assert type(cf_ans) == type(labels[idx])
            cf_correct = cf_ans == labels[idx]

            my_causal_y_preds.append(cf_ans)
            pred_correct.append(cf_correct)

        # for each seed
        # minus offset ?
        total_sample = len(labels) - offset

        # compute score for whole samples
        factual_scores.append(sum(factual_pred_correct) / total_sample)

        TE_explain.append(np.array(all_TE).mean())

        TIE_explain.append(np.array(all_TIE).mean())
        
        TIE_scores.append(sum(TIE_pred_correct) / total_sample)

        NIE_explain.append(np.array(all_NIE).mean())

        NIE_scores.append(sum(NIE_pred_correct) / total_sample)

        INTmed_explain.append(np.array(all_INTmed).mean())

        INTmed_scores.append(sum(INTmed_pred_correct) / total_sample)
        my_causal_query.append(sum(pred_correct) / total_sample)

        # save data for analysis
        raw_factual_correct.append(factual_pred_correct)
        raw_TIE.append(all_TIE)          
        
        # F1 score
        for x_f1, x_y_preds in zip(
            [factual_f1,TIE_f1, NIE_f1, INTmed_f1, my_causal_f1],
            [factual_y_preds,tie_y_preds, nie_y_preds, intmed_y_preds, my_causal_y_preds],
        ):
            f1_scores = f1_score(
                y_true=labels, y_pred=x_y_preds, average=None, labels=unique_labels
            )
            for label, f1 in zip(unique_labels, f1_scores):
                try:
                    x_f1[label].append(f1)
                except KeyError:
                    x_f1[label] = [
                        f1,
                    ]
    # MACRO F1
    factual_f1['MAF1'] = np.array(list(factual_f1.values())).mean(axis=0)
    TIE_f1['MAF1'] = np.array(list(TIE_f1.values())).mean(axis=0)
    my_causal_f1['MAF1'] = np.array(list(my_causal_f1.values())).mean(axis=0)

    factual_avg_acc =  np.array(factual_scores).mean()
    tie_avg_acc  = np.array(TIE_scores).mean()
    te_model_avg_acc = np.array(my_causal_query).mean()

    print(f"factual  score avg : {factual_avg_acc} , std : {np.array(factual_scores).std()}")
    print(f"TIE acc  avg : {tie_avg_acc} std : {np.array(TIE_scores).std()}")
    print(f"TE_model avg :  {te_model_avg_acc} , std : {np.array(my_causal_query).std()}")
    
    if DEBUG:
        print(f"TE  avg : {np.array(TE_explain).mean()} , std : {np.array(TE_explain).std()}")
        print(f"TIE avg : {np.array(TIE_explain).mean()} std : {np.array(TIE_explain).std()}")

    # return
    print(ASD(TIE_scores, factual_scores))

    print(
        "TIE F1:",
         TIE_f1,
        {"%s_mean_sd" % k: [np.mean(v), np.std(v)] for k, v in TIE_f1.items()},
    )
    
    print('checking ASD.......')
    print('TIE_MAF1:')
    print(TIE_f1['MAF1'])
    print('factual_MAF1:')
    print(factual_f1['MAF1'])
    print(ASD(TIE_f1['MAF1'], factual_f1['MAF1']))

    
    # print(my_causal_query)
    print(ASD(my_causal_query, factual_scores))
    print(
        "TE_model F1:",
        my_causal_f1,
        {"%s_mean_sd" % k: [np.mean(v), np.std(v)]
         for k, v in my_causal_f1.items()},
    )
    print('checking ASD.......')
    print('TE_model_MAF1:')
    print(my_causal_f1['MAF1'])
    print('factual_MAF1:')
    print(factual_f1['MAF1'])
    print(ASD(my_causal_f1['MAF1'], factual_f1['MAF1']))

    return factual_avg_acc, tie_avg_acc, te_model_avg_acc
    
