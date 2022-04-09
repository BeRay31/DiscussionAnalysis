import sys
import os

from numpy import sort

folder_path = sys.argv[1]
list_verbose = os.listdir(folder_path)
list_verbose.sort()

def openFileAndWriteVerbose(file):
  file1 = open(file, 'r')
  Lines = file1.readlines()
  
  # Strips the newline character
  toPrint = []
  for line in Lines:
    if "lr:" in line:
      toPrint.append(line)
  for i in range(len(toPrint)):
    print(f"Epoch {i+1}/{len(toPrint)}")
    print(toPrint[i])
  print()
  file1.close()
  return toPrint

for verbose in list_verbose:
  print(f"Verbose from {verbose}")
  verbose_dir = os.path.join(folder_path, verbose)
  openFileAndWriteVerbose(verbose_dir)

