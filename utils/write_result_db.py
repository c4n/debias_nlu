import json
import _jsonnet

import re

import psycopg2
from psycopg2.extras import Json, DictCursor
from datetime import datetime

import sys, getopt

def connect_rds():
    """Create a new database session based on configuration in PATH_CONFIG
    and return a new connection object."""
    
    PATH_CONFIG = "db_config.json"
    
    with open(PATH_CONFIG) as f:
        config = json.load(f)

    user = config['user']
    password = config['password']
    db = config['db']
    host = config['host']
    port = config['port']

    conn_str = "host={} dbname={} user={} password={}".format(host, db, user, password)
    conn = psycopg2.connect(conn_str)

    return conn

def write_to_db(info):

    ## writing logic
    sql = """INSERT INTO 
                 ci4rrl_results.nli(
        experiment_datetime,
        encoder,
        pooling_layer,
        classifer_head,
        training_strategy,
        
        num_epoch,
        debiasing_method,
        training_set,
        slurm_script,
        allennlp_jsonnet_path,
        
        allennlp_jsonnet,
        random_seed,
        numpy_seed,
        pytorch_seed,
        output_dir,
        
        exp_remarks,
        mnli_result,
        hans_e_l,
        hans_e_s,
        hans_e_c,
        
        hans_ne_l,
        hans_ne_s,
        hans_ne_c,
        naik_antonym,
        naik_numerical,
        
        naik_word_overlap,
        naik_negation,
        naik_length_mismatch,
        naik_spelling_error,
        snli_test_hard,
        
        mnli_dev_mm_hard,
        mnli_kaggle_mm_hard,
        kaushik_rp,
        kaushik_rh,
        kaushik_combined) 

        VALUES(%s,%s,%s,%s,%s,  %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s, %s,%s,%s,%s,%s);"""

    #Connection to PostGreSQL
    conn = None
    try:
        conn = connect_rds()
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute(sql, info)
        conn.commit()
        #close connection
        cur.close()

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
        #Close connection

        else:
         #Raise Error saying configuration is not valid 
            print("this configuration is not VALID!")

def clean_naik(naik_result):
    f_text = re.sub(".+{","{",naik_result.read())
    return f_text

def main(argv):
    sub_file = ''
    try:
       opts, args = getopt.getopt(argv,"hs:",["sub_file="])
    except getopt.GetoptError:
       print('write_result_db.py -s <inputfile> ')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('write_result_db.py -s <inputfile>')
          sys.exit()
       elif opt in ("-s", "--sub_file"):
          sub_file = arg



    # Opening text file
    f = open(sub_file,)
    sub_file_text = f.read()
    f.close()
    # extract info   
    # regex for sub file
    training_strategy_regex= "#psql training_strategy=(.*)"
    debiasing_method_regex= "#psql debiasing_method=(.*)"
    exp_remarks_regex= "#psql exp_remarks=(.*)"
    sub_regex= "spurious_corr/MNLI/training_config/(.+\.jsonnet)"
    model_dir_regex  = "-s\s(.+)\s--"  
    train_jsonnet = re.search(sub_regex,sub_file_text)[1] # triain config
    model_dir = re.search(model_dir_regex,sub_file_text)[1] #model dir
    if model_dir[-1]!="/":
        model_dir = model_dir +'/'
    print(train_jsonnet,model_dir)

    # metadata about the experiment
    training_strategy = re.search(training_strategy_regex,sub_file_text)[1]
    debiasing_method = re.search(debiasing_method_regex,sub_file_text)[1]
    exp_remarks = re.search(exp_remarks_regex,sub_file_text)[1]

    # Opening JSON file
    f = open(model_dir+'result.txt',)
    # returns JSON object as 
    # a dictionary
    result = json.load(f)
    f.close()

    # Opening text file
    f = open(model_dir+'hans_results.txt',)
    hans_results = f.read()
    f.close()
    
    # Opening JSON file
    f = open(model_dir+'antonym_result.txt',)
    antonym_result = json.loads(clean_naik(f))
    f.close()

    f = open(model_dir+'numerical_reasoning_result.txt',)
    numerical_result = json.loads(clean_naik(f))
    f.close()

    f = open(model_dir+'word_overlap_result.txt',)
    word_overlap_result = json.loads(clean_naik(f))
    f.close()

    f = open(model_dir+'negation_result.txt',)
    negation_result = json.loads(clean_naik(f))
    f.close()

    f = open(model_dir+'length_mismatch_result.txt',)
    length_mismatch_result = json.loads(clean_naik(f))
    f.close()

    f = open(model_dir+'spelling_result.txt',)
    spelling_result = json.loads(clean_naik(f))
    f.close()

    f = open(model_dir+'snli_hard_result.txt',)
    snli_hard_result = json.load(f)
    f.close()

    f = open(model_dir+'mnli_hard_dev_mm_result.txt',)
    mnli_hard_dev_mm_result = json.load(f)
    f.close()

    f = open(model_dir+'kaushik_rp_result.txt',)
    kaushik_rp_result = json.load(f)
    f.close()

    f = open(model_dir+'kaushik_rh_result.txt',)
    kaushik_rh_result = json.load(f)
    f.close()

    f = open(model_dir+'kaushik_combined_result.txt',)
    kaushik_combined_result = json.load(f)
    f.close()



    # regex for parsing hans results
    non_entailed_regex = r"Heuristic non-entailed results:((.*(\n|\r|\r\n)){4})"
    entailed_regex = r"Heuristic entailed results:((.*(\n|\r|\r\n)){4})"
    lexical_regex = r'lexical_overlap:\s(\d\.\d+)'
    subsequence_regex = r'subsequence:\s(\d\.\d+)'
    constituent_regex = r'constituent:\s(\d\.\d+)'

    # parse hans results
    entailed_text = re.search(entailed_regex, hans_results)[0]
    hans_e_l = re.search(lexical_regex, entailed_text)[1]
    hans_e_s = re.search(subsequence_regex, entailed_text)[1]
    hans_e_c = re.search(constituent_regex, entailed_text)[1]

    non_entailed_text = re.search(non_entailed_regex, hans_results)[0]
    hans_ne_l = re.search(lexical_regex, non_entailed_text)[1]
    hans_ne_s = re.search(subsequence_regex, non_entailed_text)[1]
    hans_ne_c = re.search(constituent_regex, non_entailed_text)[1]


    # jsonnet training config
    training_config = json.loads(_jsonnet.evaluate_file('spurious_corr/MNLI/training_config/'+train_jsonnet))


    # data to upload
    experiment_datetime = str(datetime.now()) 
    encoder = training_config['dataset_reader']['tokenizer']['model_name']
    pooling_layer = training_config['model']['seq2vec_encoder']['type']
    classifer_head  = Json(training_config['model']['feedforward'])
    training_strategy = training_strategy
    num_epoch = training_config['trainer']['num_epochs']
    debiasing_method = debiasing_method
    training_set = training_config['train_data_path']
    slurm_script = sub_file
    allennlp_jsonnet_path = train_jsonnet
    allennlp_jsonnet = Json(training_config)
    random_seed = 13370
    numpy_seed = 1337
    pytorch_seed = 133
    output_dir = model_dir
    exp_remarks = exp_remarks
    mnli_result = result['accuracy']
    hans_e_l = float(hans_e_l)
    hans_e_s = float(hans_e_s)
    hans_e_c = float(hans_e_c)
    hans_ne_l = float(hans_ne_l)
    hans_ne_s = float(hans_ne_s)
    hans_ne_c = float(hans_ne_c)    
    naik_antonym = antonym_result['accuracy']
    naik_numerical = numerical_result['accuracy'] 
    naik_word_overlap = word_overlap_result['accuracy']
    naik_negation = negation_result['accuracy']
    naik_length_mismatch = length_mismatch_result['accuracy']
    naik_spelling_error = spelling_result['accuracy']
    snli_test_hard = snli_hard_result['accuracy']
    mnli_dev_mm_hard = mnli_hard_dev_mm_result['accuracy']
    mnli_kaggle_mm_hard = 0.00,
    kaushik_rp = kaushik_rp_result['accuracy']
    kaushik_rh = kaushik_rh_result['accuracy']
    kaushik_combined = kaushik_combined_result['accuracy']

    info = (experiment_datetime,
    encoder,
    pooling_layer,
    classifer_head,
    training_strategy,
    num_epoch,
    debiasing_method,
    training_set,
    slurm_script,
    allennlp_jsonnet_path,
    allennlp_jsonnet,
    random_seed,
    numpy_seed,
    pytorch_seed,
    output_dir,
    exp_remarks,         
    mnli_result,
    hans_e_l,
    hans_e_s,
    hans_e_c,
    hans_ne_l,
    hans_ne_s,
    hans_ne_c,
    naik_antonym,
    naik_numerical,
    naik_word_overlap,
    naik_negation,
    naik_length_mismatch,
    naik_spelling_error,
    snli_test_hard,
    mnli_dev_mm_hard,
    mnli_kaggle_mm_hard,
    kaushik_rp,
    kaushik_rh,
    kaushik_combined)

    write_to_db(info)


if __name__ == "__main__":
   main(sys.argv[1:])
