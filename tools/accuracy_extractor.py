# Using readlines()
import sys
import os

from numpy import sort

def openFileAndWriteAccuracy(file):
  file1 = open(file, 'r')
  Lines = file1.readlines()
  
  # Strips the newline character
  prevLine = ""
  toPrint = []
  for line in Lines:
    if "Accuracy:" in prevLine:
      toPrint.append(round(float(line.strip()), 6))
    prevLine = line
  file1.close()
  return toPrint
  

folder_path = sys.argv[1]
list_model = sort(os.listdir(folder_path))

for model in list_model:
  # open file
  model_eval_dir = os.path.join(folder_path, model, "evaluation")
  if os.path.isdir(model_eval_dir):
    print(f"Evaluation in {model}")
    evaluation_list = os.listdir(model_eval_dir)
    evaluation_list.sort(key=lambda x: int(x[6:]))
    train = []
    dev = []
    test = []
    for eval in evaluation_list:
      evaluation_log_dir = os.path.join(model_eval_dir, eval, "log.txt")
      evalRes = openFileAndWriteAccuracy(evaluation_log_dir)
      print(eval, end=",")
      train.append(evalRes[0])
      dev.append(evalRes[1])
      test.append(evalRes[2])
    print()
    print(",".join([str(a) for a in train]))
    print(",".join([str(a) for a in dev]))
    print(",".join([str(a) for a in test]))
    print()
