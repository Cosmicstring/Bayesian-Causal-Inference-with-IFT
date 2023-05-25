import numpy as np
import nifty6 as ift

import random
import os
import json
import matplotlib.pyplot as plt

from data_processing_utilities import generate_bivar_data, generate_confounder_data



output_folder = "benchmark_tests/ConSyn/"
ground_truth_plots = "ConSyn_gt_plots/"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)
if not os.path.exists(ground_truth_plots):
    os.mkdir(ground_truth_plots)

# Define the X->Y, Y->X and X<-Z->Y models 
# through which the data is generated. Try different seeds

pairmeta = output_folder + "pairmeta.txt"

f_pairmeta = open(pairmeta, 'w')

with open('config.json', 'r') as config:
    setp = json.load(config)

# Generate X->Y
for indx in range(1,34):
    pairname = "pair{:04d}.txt".format(indx)
    output = output_folder + pairname

    print(output)
    X,Y,true_Y,noise = generate_bivar_data("X->Y", random.randrange(100), setp["real_model"]["bivariate"]["v1"])

    with open(output, 'w') as f:
        for x,y in zip(X,Y):
            f.write("{:.5e}\t{:.5e}\n".format(x,y))

    with open(ground_truth_plots + pairname, "w") as f:
        f.write("{:10s}\t{:10s}\t{:10s}\n".format("X data", "Y data", "True Y"))
        f.write("\n")
        f.write("Noise variance: {:.5e}\n".format(noise))
        f.write("\n")
        for x,y,true_y in zip(X,Y, true_Y):
                f.write("{:.5e}\t{:.5e}\t{:.5e}\n".format(x,y, true_y))        

    f_pairmeta.write("{:10s} 1 1 2 2 1\n".format(pairname[:-4]))

# Generate Y->X
for indx in range(34,66):

    pairname = "pair{:04d}.txt".format(indx)
    output = output_folder + pairname

    print(output)
    Y, X, true_X, noise = generate_bivar_data("Y->X", random.randrange(100), setp["real_model"]["bivariate"]["v1"])

    with open(output, 'w') as f:
        for x,y in zip(X,Y):
            f.write("{:.5e}\t{:.5e}\n".format(x,y))

    with open(ground_truth_plots + pairname, "w") as f:
        f.write("{:10s}\t{:10s}\t{:10s}\n".format("X data", "Y data", "True Y"))
        f.write("\n")
        f.write("Noise variance: {:.5e}\n".format(noise))
        f.write("\n")
        for x,y,true_x in zip(Y,X,true_X):
                f.write("{:.5e}\t{:.5e}\t{:.5e}\n".format(x,y, true_x))

    f_pairmeta.write("{:10s} 2 2 1 1 1\n".format(pairname[:-4]))

for indx in range(66,101):
    pairname = "pair{:04d}.txt".format(indx)
    output = output_folder + pairname

    print(output)
    X,Y, Z, pdf_Z, f_X, f_Y, var_X, var_Y = generate_confounder_data(random.randrange(100), setp, version="v1")

    with open(output, 'w') as f:
        for x,y in zip(X,Y):
            f.write("{:.5e}\t{:.5e}\n".format(x,y))

    with open(ground_truth_plots + pairname, "w") as f:
        f.write("{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\t{:10s}\n".format("X data", "Y data", "True X", "True Y", "Z", "pdf of Z"))
        f.write("\n")
        f.write("Noise variances: {:.5e}\t{:.5e}\n".format(var_X, var_Y))
        f.write("\n")
        for x,y,true_x, true_y, z, pdf_z in zip(X,Y, f_X, f_Y, Z, pdf_Z):
                f.write("{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\t{:.5e}\n".format(x,y, true_x, true_y, z, pdf_z))


    f_pairmeta.write("{:10s} 1 2 3 3 1\n".format(pairname[:-4]))

f_pairmeta.close()