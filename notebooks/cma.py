import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.special import expit
import fuse
import causal_utils 
import glob
from kl_general import sharpness_correction


def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"
    
def get_ans(ans,test_set):
    if test_set =='hans':
        if ans == 0:
            return 'entailment'
        else:
            return 'non-entailment' 
    else:
        key= {0:"entailment",1:"contradiction",2:"neutral"}
        return key[ans]
        
    
# model_path='/raid/can/nli_models/reweight_utama_github/'
# task='nli'
# data_path='/ist/users/canu/debias_nlu/data/' + task + '/'
# fusion = fuse.sum_fuse
# test_set='hans'


def get_c(data_path,model_path,fusion,avg):
    df_bias_dev = pd.read_json(
    data_path+'dev_prob_korn_lr_overlapping_sample_weight_3class.jsonl', lines=True)
    bias_dev_score = [b for b in df_bias_dev['bias_probs']]
    bias_dev_score = np.array(bias_dev_score)
    y1m0_dev = fusion(avg, bias_dev_score)
    df_bert_dev = pd.read_json(model_path+'raw_m.jsonl', lines=True)
    y1m1prob_dev = []
    for p, h in zip(df_bert_dev['probs'], y1m0_dev):
        new_y1m1 = fusion(np.array(p), h)
        y1m1prob_dev.append(new_y1m1)
    c = sharpness_correction(bias_dev_score, y1m1prob_dev) 
    return c

def report_CMA(model_path,task,data_path,test_set,fusion,correction=False):
    df_bias_model = pd.read_json(data_path+test_set+'_prob_korn_lr_overlapping_sample_weight_3class.jsonl', lines=True)
    bias_only_scores=[b for b in df_bias_model['bias_probs'] ]
    bias_only_scores=np.array(bias_only_scores)


    to_glob = model_path + task + '/*/'


    seed_path=glob.glob(to_glob)
    TE_explain = []
    TIE_explain = []
    factual_scores = []
    TIE_scores = []
    NIE_explain = []
    NIE_scores = []
    INTmed_explain = []
    INTmed_scores = []
    my_causal_query = []

    # get avg score
    for seed_idx in range(len(seed_path)):
        df = pd.read_json(seed_path[seed_idx]+'raw_train.jsonl', lines=True)
        list_probs = []
        for i in df['probs']:
            list_probs.extend(i)
        x=np.array(list_probs)
        avg=np.average(x,axis=0)
        if correction:
            avg = get_c(data_path,seed_path[seed_idx],fusion,avg)
        # fusion to create y1m0
        y1m0prob=fusion(bias_only_scores,avg)

        # get score of the model on a challenge set
        if test_set == 'hans':
            result_path=seed_path[seed_idx]+'normal/'
            print(result_path)
            df_bert = pd.read_json(result_path+'hans_result.jsonl', lines=True)
        elif test_set == 'test':
            result_path=seed_path[seed_idx]+'/'
            df_bert = pd.read_json(result_path+'raw_mm.jsonl', lines=True)
        elif test_set == 'dev':
            result_path=seed_path[seed_idx]+'/'
            df_bert = pd.read_json(result_path+'raw_m.jsonl', lines=True)            
        # y1m1
        y1m1prob = []
        for p,h in zip(bias_only_scores,df_bert['probs']):
            new_y1m1 = fusion(np.array(p),h)
            y1m1prob.append(new_y1m1)

        debias_scores = []
        for p,b in zip(y1m1prob,y1m0prob):
            debias_scores.append(p-b)    

#         {0:"entailment",1:"contradiction",2:"neutral"}
        labels=df_bias_model['gold_label']
        offset = 0 
        if '-' in df_bias_model['gold_label'].value_counts():
            # no ground truth
            offset=df_bias_model['gold_label'].value_counts()['-'] 
        # CMA
        import pickle
        modelname='mnli_lr_model.sav'
        loaded_model = pickle.load(open(modelname, 'rb'))
        x0=loaded_model.predict_proba(np.array([[0,0,0.41997876976119086]]))
        m0=avg
        y0m0=fusion(x0,m0)      
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
        for i in range(len(labels)): 
            y1m1 = y1m1prob[i]
            y1m0 = y1m0prob[i]
            TE = y1m1 - y0m0
            NDE = y1m1 - y0m0
            y0m1= fusion(x0,np.array(df_bert['probs'][i]) )
            TIE = y1m1 - y1m0
            NIE = y0m1 - y0m0
            INTmed = TIE - NIE
            # factual
            factual_ans = np.argmax(df_bert['probs'][i])
            factual_ans = get_ans(factual_ans,test_set)
            factual_correct = factual_ans==labels[i]   
            factual_pred_correct.append(factual_correct)
            # TIE
            TIE_ans = np.argmax(TIE)
            TIE_ans = get_ans(TIE_ans,test_set)
            TIE_correct = TIE_ans==labels[i]  
            TIE_pred_correct.append(TIE_correct)
            # INTmed
            INTmed_ans = np.argmax(INTmed[0])
            INTmed_ans = get_ans(INTmed_ans,test_set)
            INTmed_correct = INTmed_ans==labels[i]  
            INTmed_pred_correct.append(INTmed_correct)
            # NIE
            NIE_ans = np.argmax(NIE[0])
            NIE_ans = get_ans(NIE_ans,test_set)
            NIE_correct = NIE_ans==labels[i]  
            NIE_pred_correct.append(NIE_correct)    

            # save
            all_NDE.append(NDE[0][0])
            all_NIE.append(NIE[0][0])
            all_TIE.append(TIE[0])
            all_TE.append(TE[0][0])
            all_INTmed.append((INTmed[0][0]))
            if  (TIE[0]/TE[0][0])<9999999:
                cf_ans = np.argmax(np.array(df_bert['probs'][i]-bias_only_scores[i]))
                cf_ans = get_ans(cf_ans,test_set)  
                cf_correct = cf_ans==labels[i]
            else:
        #         print(cf_ans)
                cf_correct = factual_ans ==labels[i]
            pred_correct.append(cf_correct)

        #     np.array(df_bert['probs'][i]-y1m0prob)
        #     labels[i]
        total_sample = len(labels)- offset
        factual_scores.append(sum(factual_pred_correct)/total_sample)
        TE_explain.append(np.array(all_TE).mean())
        TIE_explain.append(np.array(all_TIE).mean())
        TIE_scores.append(sum(TIE_pred_correct)/total_sample)
        NIE_explain.append(np.array(all_NIE).mean())
        NIE_scores.append(sum(NIE_pred_correct)/total_sample)    
        INTmed_explain.append(np.array(all_INTmed).mean())
        INTmed_scores.append(sum(INTmed_pred_correct)/total_sample)      
        my_causal_query.append(sum(pred_correct)/total_sample)
        print(np.array(all_TIE).mean(),(np.array(all_TIE).std()))
        print(np.array(all_INTmed).mean(),(np.array(all_INTmed).std()))
        print(np.array(all_NIE).mean(),(np.array(all_NIE).std()))

    print('factual score:')
    print(factual_scores,np.array(factual_scores).mean(),np.array(factual_scores).std()) 
    print("TE:")
    print(TE_explain,np.array(TE_explain).mean(),np.array(TE_explain).std()) 
    print("TIE:")
    print(TIE_explain,np.array(TIE_explain).mean(),np.array(TIE_explain).std())
    print("TIE acc:")
    print(TIE_scores,np.array(TIE_scores).mean(),np.array(TIE_scores).std())
    print("NIE:")
    print(NIE_explain,np.array(NIE_explain).mean(),np.array(NIE_explain).std())
    print("NIE acc:")
    print(NIE_scores,np.array(NIE_scores).mean(),np.array(NIE_scores).std())
    print("INTmed:")
    print(INTmed_explain,np.array(INTmed_explain).mean(),np.array(INTmed_explain).std())
    print("INTmed acc:")
    print(INTmed_scores,np.array(INTmed_scores).mean(),np.array(INTmed_scores).std())
    print("my query:")
    print(my_causal_query,np.array(my_causal_query).mean(),np.array(my_causal_query).std())