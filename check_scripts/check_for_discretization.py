import numpy as np
import os

from data_processing_utilities import neglected_files

import matplotlib.pyplot as plt


input_folder = "benchmark_tests/tcep_no_des/"

files = os.listdir(input_folder)
files.sort()

pairmeta = "benchmark_tests/tcep/pairmeta.txt"

true_direction = {}
with open(pairmeta, "r") as f:
    for line in f:
        _line = line.split()
        filename = _line[0]
        if not (filename in neglected_files["tcep_no_des"]):
            tmp = [float(x) for x in _line[1:]]
            # Assuming here there are no directions except 'X->Y'
            # and 'Y->X'
            if \
                tmp[0]==1 and tmp[1] ==1 and tmp[2]==2 and tmp[3]==2:
                true_direction["pair" + filename] = 'X->Y'
            elif \
                tmp[0]==2 and tmp[1]==2 and tmp[2]==1 and tmp[3]==1:
                true_direction["pair" + filename] = 'Y->X'

discretized = []
total_count = 0
for file in files:
    print(file)
    if not ((file[:8] + '.txt') in neglected_files["tcep_no_des"]):
        total_count+=1

        data = np.loadtxt(input_folder + file, unpack=True)
        X,Y = data[0,:], data[1,:]

        (unique_X, idx_X) = np.unique(X, return_index=True)
        (unique_Y, idx_Y) = np.unique(Y, return_index=True)

        if true_direction[file[:8]] == "X->Y":
            if (unique_X.size < int(0.5 * len(X))):
                discretized.append(file[:-4])
        
        if true_direction[file[:8]] == "Y->X":
            if (unique_Y.size < int(0.5 * len(Y))):
                discretized.append(file[:-4])

print(total_count)
print(len(discretized))
print(discretized)
print(len(discretized) / total_count)
