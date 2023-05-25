import numpy as np

import matplotlib.pyplot as plt

path_in = "../benchmark_tests/tcep/pair0025.txt"

X,Y = np.loadtxt(path_in, unpack=True)

plt.plot(X,Y, 'ko', alpha=.5)
plt.xlabel(r'Cement $[\rm{kg\,\,m^{-3}}]$')
plt.ylabel(r'Concrete compressive strength $[\rm{MPa}]$')

plt.savefig("pair0025.pdf")