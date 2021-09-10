# Ref:https://machinelearningmastery.com/mcnemars-test-for-machine-learning/
from statsmodels.stats.contingency_tables import mcnemar
import sys, getopt
import pickle

def main(argv):
    baseline_file = ''
    target_file = ''
    
    try:
       opts, args = getopt.getopt(argv,"hb:t:",["help","baseline_file=","target_file="])
    except getopt.GetoptError:
       print('mcnemar_test.py -b <baselinefile> -t <targetfile>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('mcnemar_test.py -b <baselinefile> -t <targetfile>')
          sys.exit()
       elif opt in ("-b", "--baseline_file"):
          baseline_file = arg
       elif opt in ("-t", "--target_file"):
          target_file = arg        
    # load files here
    baseline_result = pickle.load( open( baseline_file, "rb" ) )
    target_result = pickle.load( open( target_file, "rb" ) )

    # count for contingency table
    keys = baseline_result.keys()
    yy = 0 # yes/yes
    yn = 0 # yes/no
    ny = 0 # no/yes
    nn = 0 # no/no
    for key in keys:
        if baseline_result[key] == 'yes' and target_result[key] == 'yes':
            yy += 1 
        elif baseline_result[key] == 'yes' and target_result[key] == 'no':
            yn += 1 
        elif baseline_result[key] == 'no' and target_result[key] == 'yes':
            ny += 1 
        elif baseline_result[key] == 'no' and target_result[key] == 'no':
            nn += 1

    # define contingency table
    table = [[yy, yn],
            [ny, nn]]
    # table = [[4, 2],
    #         [1, 3]]
    # calculate mcnemar test
    result = mcnemar(table, exact=False)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

if __name__ == "__main__":
    main(sys.argv[1:])
      