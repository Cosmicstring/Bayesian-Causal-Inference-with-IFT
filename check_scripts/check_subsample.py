import numpy as np
import os

import matplotlib.pyplot as plt

from data_processing_utilities import _readin_data, _normalize

sub_path = 'benchmark_tests/subsampled/'
tcep_path = 'benchmark_tests/tcep_no_des/'

file_sub = os.listdir(sub_path)
file_tcep = os.listdir(tcep_path)

scale = (0,1)

for file in file_sub:
    print(file)

    X_s, Y_s = _readin_data(file, sub_path)

    X,Y = _readin_data(file[:8] + '.txt', tcep_path)
    X,Y = _normalize(X, Y, scale)

    plt.plot(X,Y, 'bo', markersize=5)
    plt.plot(X_s,Y_s, 'rx', markersize=5)
    plt.show()