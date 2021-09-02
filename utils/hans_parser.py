import sys, getopt
import jsonlines
import datetime
import time

def main(argv):
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print('hans_parser.py -i <inputfile> -o <outputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print('hans_parser.py -i <inputfile> -o <outputfile>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
   print('Input file is ', inputfile)
   print('Output file is ', outputfile)
   with jsonlines.open(inputfile) as reader:
       with open(outputfile, 'w') as writer:
           writer.write("pairID,gold_label\n")
           for idx, obj in enumerate(reader):
               writer.write("ex"+str(idx)+","+obj['label']+"\n")      
if __name__ == "__main__":
   main(sys.argv[1:])