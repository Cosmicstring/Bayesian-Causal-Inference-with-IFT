import numpy as np
import os
import random

from data_processing_utilities import _readin_data, _normalize

scale = (0,1)

path_in = "benchmark_tests/SIM-G/"
path_out = "benchmark_tests/SIM-G_subsampled/"

N_samples = 1000

files = os.listdir(path_in)
files.sort()

for _file in files:    
    X,Y = _readin_data(_file, path_in) 
    X,Y = _normalize(X,Y, scale)

    if len(X) > N_samples:
        print(_file)
        data1 = list(X)               
        data2 = list(Y)
        X_sub = []
        Y_sub = []
        for i in range(N_samples):
            index = random.randrange(len(data1))
            elem1 = data1[index]
            elem2 = data2[index]
            
            del data1[index]
            del data2[index]
            X_sub.append(elem1)
            Y_sub.append(elem2)
         
        p_out = path_out + _file[:-4] + "_subsampled.txt"
        with open(p_out, "w") as f:
            for x,y in zip(X_sub,Y_sub):
                f.write("{:.10e} {:.10e}\n".format(x,y))
    else:
        cmd = "cp " + path_in + _file + " " + path_out
        os.popen(cmd)
