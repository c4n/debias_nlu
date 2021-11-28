import jsonlines
import sys, getopt


def main(argv):
    output_path = ''

    
    try:
       opts, args = getopt.getopt(argv,"ht:p:o:",["help","train_path=","pred_path=","output_path="])
    except getopt.GetoptError:
       print('create_distill_train_set.py -t <train_path> -p <pred_path> -o <output_path>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('create_distill_train_set.py -t <train_path> -p <pred_path> -o <output_path>')
          sys.exit()
       elif opt in ("-t", "--train_path"):
          train_path = arg    
       elif opt in ("-p", "--pred_path"):
          pred_path = arg               
       elif opt in ("-o", "--output_path"):
          output_path = arg

    # load files here
    out = []
    with jsonlines.open(train_path) as f1,jsonlines.open(pred_path) as f2:
        if len(list(f1.iter())) != len(list(f2.iter())):
           raise Exception('Length Mismatch: Please check the files')

    with jsonlines.open(train_path) as f1,jsonlines.open(pred_path) as f2:
        for line, line_pred in zip(f1.iter(),f2.iter()):
            line['logits'] = line_pred['logits']
            line['distill_probs'] = line_pred['probs']
            out.append(line)

    with jsonlines.open(output_path, 'w') as writer:
        writer.write_all(out)

if __name__ == "__main__":
    main(sys.argv[1:])
      