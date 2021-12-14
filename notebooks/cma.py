import pandas as pd
import numpy as np
from scipy.special import softmax
from scipy.special import expit
import fuse
import causal_utils 
import glob


def format_label(label):
    if label == "entailment":
        return "entailment"
    else:
        return "non-entailment"
    
def get_ans(ans):
    if ans == 0:
        return 'entailment'
    else:
        return 'non-entailment' 
        
    
model_path='/raid/can/nli_models/korn_reweight/'
task='nli'
data_path='/ist/users/canu/debias_nlu/data/' + task + '/'
fusion = fuse.sum_fuse

df_hans = pd.read_json(data_path+'hans_prob_korn_lr_overlapping_sample_weight_3class.jsonl', lines=True)
hans_score=[b for b in df_hans['bias_probs'] ]
hans_score=np.array(hans_score)


to_glob = model_path + task + '/*/'


seed_path=glob.glob(to_glob)
TE_explain = []
TIE_explain = []
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
    # fusion to create y0m0
    bias_score=fusion(avg,hans_score)

    # get score of the model on a challenge set
    result_path=seed_path[seed_idx]+'normal/'
    df_bert = pd.read_json(result_path+'hans_result.jsonl', lines=True)

    # y1m1
    y1m1prob = []
    for p,h in zip(df_bert['probs'],hans_score):
        new_y1m1 = fusion(np.array(p),h)
        y1m1prob.append(new_y1m1)

    debias_scores = []
    for p,b in zip(y1m1prob,bias_score):
        debias_scores.append(p-b)    

    key = {0:"entailment",1:"contradiction",2:"neutral"}
    labels = []
    for i in debias_scores:
        labels.append(key[np.argmax(i)])
    df_bert['debias_label']=labels    


    # get hans score
    text_ans=""
    for idx, obj in enumerate(df_bert['label']):
        text_ans = text_ans + "ex"+str(idx)+","+obj+"\n"   

    text_ans_debias=""
    for idx, obj in enumerate(df_bert['debias_label']):
        text_ans_debias = text_ans_debias + "ex"+str(idx)+","+obj+"\n"          



    guess_dict = {}
    for line in text_ans.split("\n"):
        if len(line)>1:
            parts = line.strip().split(",")
            guess_dict[parts[0]] = format_label(parts[1])

    # # guess_dict_debias = {}
    # # for line in text_ans_debias.split("\n"):
    # #     if len(line)>1:
    # #         parts = line.strip().split(",")
    # #         guess_dict_debias[parts[0]] = format_label(parts[1])  

    labels,baseline_avg=causal_utils.get_heur(guess_dict)        
    # labels,debias_avg=causal_utils.get_heur(guess_dict_debias)


    # CMA
    import pickle
    modelname='mnli_lr_model.sav'
    loaded_model = pickle.load(open(modelname, 'rb'))
    x0=loaded_model.predict_proba(np.array([[0,0,0.41997876976119086]]))
    m0=avg
    y0m0=fusion(x0,m0)      

    factual_pred_correct = []
    TIE_pred_correct = []
    NIE_pred_correct = []
    INTmed_pred_correct = []
    pred_correct = []
    all_TE = []
    all_TIE = []
    all_NIE = []
    all_NDE = []
    all_INTmed = []
    for i in range(len(labels)): 
        y1m1 = y1m1prob[i]
        y1m0 = bias_score[i]
        TE = y1m1 - y0m0
        NDE = bias_score[i] - y0m0
        y0m1= fusion(x0,np.array(df_bert['probs'][i]) )
        TIE = y1m1 - y1m0
        NIE = y0m1 - y0m0
        INTmed = TIE - NIE
        # factual
        factual_ans = np.argmax(df_bert['probs'][i])
        factual_ans = get_ans(factual_ans)
        factual_correct = factual_ans==labels[i]   
        factual_pred_correct.append(factual_correct)
        # TIE
        TIE_ans = np.argmax(TIE)
        TIE_ans = get_ans(TIE_ans)
        TIE_correct = TIE_ans==labels[i]  
        TIE_pred_correct.append(TIE_correct)
        # INTmed
        INTmed_ans = np.argmax(INTmed[0])
        INTmed_ans = get_ans(INTmed_ans)
        INTmed_correct = INTmed_ans==labels[i]  
        INTmed_pred_correct.append(INTmed_correct)
        # NIE
        NIE_ans = np.argmax(NIE[0])
        NIE_ans = get_ans(NIE_ans)
        NIE_correct = NIE_ans==labels[i]  
        NIE_pred_correct.append(NIE_correct)    

        # save
        all_NDE.append(NDE[0][0])
        all_NIE.append(NIE[0][0])
        all_TIE.append(TIE[0])
        all_TE.append(TE[0][0])
        all_INTmed.append((INTmed[0][0]))
        if  (TIE[0]/TE[0][0])<9999999:
            cf_ans = np.argmax(np.array(df_bert['probs'][i]-hans_score[i]))
            cf_ans = get_ans(cf_ans)  
            cf_correct = cf_ans==labels[i]
        else:
    #         print(cf_ans)
            cf_correct = factual_ans ==labels[i]
        pred_correct.append(cf_correct)

    #     np.array(df_bert['probs'][i]-bias_score)
    #     labels[i]
    TE_explain.append(np.array(all_TE).mean())
    TIE_explain.append(np.array(all_TIE).mean())
    TIE_scores.append(sum(TIE_pred_correct)/len(labels))
    NIE_explain.append(np.array(all_NIE).mean())
    NIE_scores.append(sum(NIE_pred_correct)/len(labels))    
    INTmed_explain.append(np.array(all_INTmed).mean())
    INTmed_scores.append(sum(INTmed_pred_correct)/len(labels))      
    my_causal_query.append(sum(pred_correct)/len(labels))
    print(np.array(all_TIE).mean(),(np.array(all_TIE).std()))
    print(np.array(all_INTmed).mean(),(np.array(all_INTmed).std()))
    print(np.array(all_NIE).mean(),(np.array(all_NIE).std()))
    
print("TE:")
print(TE_explain,np.array(TE_explain).mean(),np.array(TE_explain).std()) 
print("TIE:")
print(TIE_explain,np.array(TIE_explain).mean(),np.array(TIE_explain).std())
print(TIE_scores,np.array(TIE_scores).mean(),np.array(TIE_scores).std())
print("NIE:")
print(NIE_explain,np.array(NIE_explain).mean(),np.array(NIE_explain).std())
print(NIE_scores,np.array(NIE_scores).mean(),np.array(NIE_scores).std())
print("INTmed:")
print(INTmed_explain,np.array(INTmed_explain).mean(),np.array(INTmed_explain).std())
print(INTmed_scores,np.array(INTmed_scores).mean(),np.array(INTmed_scores).std())
print("my query:")
print(my_causal_query,np.array(my_causal_query).mean(),np.array(my_causal_query).std())