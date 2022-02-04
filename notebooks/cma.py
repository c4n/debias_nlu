import os
from typing import Callable, List, Union, Tuple

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
    "mnli_train": "nli/train_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    "mnli_dev_mm": "nli/test_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    "mnli_hans": "nli/hans_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    "fever_train": "weighted_fever.train.jsonl",
    "fever_dev": "weighted_fever.dev.jsonl",
    "fever_sym1": "weighted_fever_symmetric_v0.1.test.jsonl",
    "fever_sym2": "weighted_fever_symmetric_v0.2.test.jsonl",
    # "weighted_qqp.train.jsonl",
    "qqp_train": "qqp_train_overlap_only_bias_weighted.jsonl",
    # "weighted_qqp.dev.jsonl",
    "qqp_dev": "qqp_dev_overlap_only_bias_weighted.jsonl",
    # "weighted_paws.dev_and_test.jsonl",
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
    fusion: Callable[[PROB_T], PROB_T],
    x0: PROB_T,
    bias_val_pred_file: str = "dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    model_val_pred_file: str = "raw_m.jsonl",
    bias_probs_key: str = "bias_probs",
    model_probs_key: str = "probs",
    config: dict = ESTIMATE_C_DEFAULT_CONFIG,
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
    ya1x1prob_dev = []
    for p, h in zip(df_bert_dev[model_probs_key], bias_dev_score):
        new_ya1x1 = fusion(np.array(p), h)
        ya1x1prob_dev.append(new_ya1x1)
    c = sharpness_correction(bias_dev_score, ya1x1prob_dev, config=config)
    n_labels = bias_dev_score[0].shape[0]
    c = c*np.ones(n_labels)
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
    bias_probs_key: str = "bias_probs",
    ground_truth_key: str = "gold_label",
    model_pred_method: Callable[[List[float]], List[float]] = _default_model_pred,
    bias_val_pred_file: str = "dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl",
    model_val_pred_file: str = "raw_m.jsonl",
    entropy_threshold: float = -9999,
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
    print(os.path.join(data_path, BIAS_MODEL_DICT[test_set]))
    df_bias_model = pd.read_json(
        os.path.join(data_path, BIAS_MODEL_DICT[test_set]), lines=True
    )
    a1 = [b for b in df_bias_model[bias_probs_key]]  # prob for all classes
    a1 = np.array(a1)  # [N_batch, n_class]
    n_labels = a1.shape[1]

    # get a list of all seed dir
    if not seed_path:
        to_glob = model_path + task + "/*/"
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
        print(
            os.path.join(
                seed_path[seed_idx], BERT_MODEL_RESULT_DICT[TASK2TRAIN_DICT[task]]
            )
        )
        # df_val = pd.read_json(
        #     os.path.join(
        #         seed_path[seed_idx], BERT_MODEL_RESULT_DICT[TASK2TRAIN_DICT[task]]
        #     ),
        #     lines=True,
        # )
        # list_probs = []
        # for i in df_val["probs"]:
        #     list_probs.append(i)
        # train_pred_results = np.array(list_probs)
        # x0 = np.average(train_pred_results, axis=0)
        # n_labels = x0.shape[0]
        x0 = (1/n_labels) * np.ones(n_labels)
        te_correction =  np.ones(n_labels)
        if correction:
            x0 = get_c(
                data_path=os.path.join(data_path, task),
                model_path=seed_path[seed_idx],
                fusion=fusion,
                x0=x0,
                bias_val_pred_file=bias_val_pred_file,
                model_val_pred_file=model_val_pred_file,
                bias_probs_key=bias_probs_key,
                model_probs_key="probs",
                config=estimate_c_config,
            )
            te_correction = (1/n_labels) * np.ones(n_labels)
            te_correction = get_c_te(
                data_path=os.path.join(data_path, task),
                model_path=seed_path[seed_idx],
                fusion=fusion,
                x0=te_correction,
                bias_val_pred_file=bias_val_pred_file,
                model_val_pred_file=model_val_pred_file,
                bias_probs_key=bias_probs_key,
                model_probs_key="probs",
                config=estimate_c_te_config,
            )
        # fusion to create ya1x0
        ya1x0prob = fusion(a1, x0)

        # get score of the model on a challenge set
        df_bert = pd.read_json(
            os.path.join(seed_path[seed_idx],
                         BERT_MODEL_RESULT_DICT[test_set]),
            lines=True,
        )

        # ya1x1
        ya1x1prob = []
        x1 = df_bert["probs"]
        for b, p in zip(a1, x1):
            new_ya1x1 = fusion(np.array(b), p)
            ya1x1prob.append(new_ya1x1)

        debias_scores = []
        for factual, counterfactual in zip(ya1x1prob, ya1x0prob):
            debias_scores.append(factual - counterfactual)

        # {0:"entailment",1:"contradiction",2:"neutral"}
        labels = df_bias_model[ground_truth_key]
        unique_labels = labels.unique().tolist()
        print("unique_labels: ", unique_labels)

        # to offset samples with no ground truth from accuracy calculation
        offset = 0
        if "-" in df_bias_model[ground_truth_key].value_counts():
            # no ground truth
            offset = df_bias_model[ground_truth_key].value_counts()["-"]
        # CMA
        # a0 = model_pred_method(_input=input_a0)  # input_a0?
        a0 = (1/n_labels) * np.ones(n_labels)
        # print("input_a0: ", input_a0)
        # print("a0: ", a0)
        ya0x0 = fusion(a0, x0)
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
        for i in range(len(labels)):
            ya1x1 = ya1x1prob[i]
            ya1x0 = ya1x0prob[i]

            TE = ya1x1 - ya0x0
            NDE = ya1x0 - ya0x0
            ya0x1 = fusion(a0, np.array(x1[i]))
            TIE = ya1x1 - ya1x0
            NIE = ya0x1 - ya0x0
            INTmed = TIE - NIE
            # factual ## Todo: wrong index for FEVER
            factual_ans = np.argmax(x1[i])
            factual_ans = get_ans(factual_ans, test_set)
            assert type(factual_ans) == type(labels[i])
            # print(x1[i], factual_ans, labels[i])
            # print("df_bert.iloc[i]: ", df_bert.iloc[i])
            # print("df_bias_model.iloc[i]: ", df_bias_model.iloc[i])
            factual_y_preds.append(factual_ans)
            factual_correct = factual_ans == labels[i]
            factual_pred_correct.append(factual_correct)
            # TIE
            TIE_ans = np.argmax(TIE)
            TIE_ans = get_ans(TIE_ans, test_set)
            assert type(TIE_ans) == type(labels[i])
            tie_y_preds.append(TIE_ans)
            TIE_correct = TIE_ans == labels[i]
            TIE_pred_correct.append(TIE_correct)
            # INTmed
            INTmed_ans = np.argmax(INTmed)
            INTmed_ans = get_ans(INTmed_ans, test_set)
            assert type(INTmed_ans) == type(labels[i])
            intmed_y_preds.append(INTmed_ans)
            INTmed_correct = INTmed_ans == labels[i]
            INTmed_pred_correct.append(INTmed_correct)
            # NIE
            NIE_ans = np.argmax(NIE)
            NIE_ans = get_ans(NIE_ans, test_set)
            assert type(NIE_ans) == type(labels[i])
            nie_y_preds.append(NIE_ans)
            NIE_correct = NIE_ans == labels[i]
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
                df_bert["probs"][i] *
                np.log(df_bert["probs"][i]) / np.log(n_labels)
            )
            if entropy > entropy_threshold:
                cf_ans = np.argmax(np.array(x1[i] - te_correction*a1[i]))
                # cf_ans = np.argmax(np.array(x1[i]  - a1[i]))
                cf_ans = get_ans(cf_ans, test_set)
                assert type(cf_ans) == type(labels[i])
                cf_correct = cf_ans == labels[i]
            else:
                assert type(factual_ans) == type(labels[i])
                cf_correct = factual_ans == labels[i]
            my_causal_y_preds.append(cf_ans)
            pred_correct.append(cf_correct)

        total_sample = len(labels) - offset
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
    factual_f1['MAF1']=np.array(list(factual_f1.values())).mean(axis=0)
    TIE_f1['MAF1']=np.array(list(TIE_f1.values())).mean(axis=0)
    my_causal_f1['MAF1']=np.array(list(my_causal_f1.values())).mean(axis=0)

    print("factual score:")
    print(factual_scores)
    print(np.array(factual_scores).mean(), np.array(factual_scores).std())

    print("TE:")
    print(np.array(TE_explain).mean(), np.array(TE_explain).std())

    print("TIE:")
    print(np.array(TIE_explain).mean(), np.array(TIE_explain).std())
    print("TIE acc:")
    print(TIE_scores)
    print(np.array(TIE_scores).mean(), np.array(TIE_scores).std())
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
    
    print("NIE:")
    print(np.array(NIE_explain).mean(), np.array(NIE_explain).std())
    print("NIE acc:")
    print(NIE_scores)
    print(np.array(NIE_scores).mean(), np.array(NIE_scores).std())

    print("my query:")
    print(my_causal_query)
    print(
        np.array(my_causal_query).mean(),
        np.array(my_causal_query).std(),
    )
    print(ASD(my_causal_query, factual_scores))
    print(
        "my query F1:",
        my_causal_f1,
        {"%s_mean_sd" % k: [np.mean(v), np.std(v)]
         for k, v in my_causal_f1.items()},
    )
    print('checking ASD.......')
    print('my_causal_MAF1:')
    print(my_causal_f1['MAF1'])
    print('factual_MAF1:')
    print(factual_f1['MAF1'])
    print(ASD(my_causal_f1['MAF1'], factual_f1['MAF1']))
    
    if return_raw:
        # Return raw values
        return raw_factual_correct, raw_TIE