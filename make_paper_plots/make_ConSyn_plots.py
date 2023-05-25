import numpy as np
import nifty6 as ift

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import json

import glob
import sys

sys.path.append("/home/joka/Documents/Andrija_Faks/WiSe_19_20/IFT/BCI/andrija-kostic/Bayesian_Causal_Inference/implementation/")

from model_utilities import load_KL_sample, load_KL_position
from data_processing_utilities import generate_confounder_data, _normalize

from causal_model import Causal_Model
from select_model import select_model


X,Y, f_X, f_Y, Z, pdf_Z,  = np.loadtxt('ConSyn_gt_plots/pair0087.txt', unpack=True, skiprows=4)

reduce_noise=1

if reduce_noise:

    N_X = ift.ScalingOperator(ift.makeDomain(ift.UnstructuredDomain(X.shape)), 1e-5)
    N_Y = ift.ScalingOperator(ift.makeDomain(ift.UnstructuredDomain(Y.shape)), 1e-5)

    data_X = f_X + N_X.draw_sample_with_dtype(np.float64).val
    data_Y = f_Y + N_Y.draw_sample_with_dtype(np.float64).val

    with open("pair0087_new.txt", "w") as f:
        for x, y in zip(data_X, data_Y):
            f.write("{:.5e}\t{:.5e}\n".format(x,y))

exit()

#pair0087 is a good example
if 0:

    with open('ConSyn_gt_plots/config_gt_params.json', 'r') as config:
        setp = json.load(config)

    min_Y, max_Y = min(Y), max(Y)
    min_X, max_X = min(X), max(X)

    fig, ax = plt.subplots(2,2, figsize=(10,10))

    hist_Z, edges = np.histogram(Z, bins=pdf_Z.size)
    x_z = 0.5*(edges[1:] + edges[:-1])
    ax[0,0].plot(x_z, hist_Z, 'k-', alpha=.5, linewidth=.5, label='Hist. of sampled Z values')

    x_pdf = np.linspace(0, 1, num=pdf_Z.size)
    ax[0,0].plot(x_pdf, pdf_Z/max(pdf_Z), 'k-', linewidth=2, alpha=1.0, label=r'$P_z$')
    ax[0,0].set_xlim(-.05,1.05)
    ax[0,0].set_xlabel(r'$z$')
    ax[0,0].set_ylabel('Counts')
    ax[0,0].legend(loc='best')


    indx = Z.argsort()
    ax[0,1].plot(f_X[indx], Z[indx], 'k-', linewidth=2., label=r'$f_X(Z)$')
    ax[0,1].set_ylim(-.05,1.05)
    ax[0,1].set_xlim(min_X-0.05,max_X+0.05)
    ax[0,1].set_xlabel(r'$f_X(Z)$')
    ax[0,1].set_ylabel(r'$Z$')
    ax[0,1].legend(loc='best')

    ax[1,0].plot(Z[indx], f_Y[indx], 'k-', linewidth=2., label=r'$f_Y(Z)$')
    ax[1,0].set_ylim(min_Y-0.05,max_Y+0.05)
    ax[1,0].set_xlim(-.05,1.05)
    ax[1,0].set_ylabel(r'$f_Y(Z)$')
    ax[1,0].set_xlabel(r'$Z$')
    ax[1,0].legend(loc='best')

    ax[1,1].plot(X,Y, 'ko', alpha=.1, label="Observed data")
    ax[1,1].plot(f_X[indx],f_Y[indx], 'r-', linewidth=2., alpha=1.0, label="Ground truth")
    ax[1,1].set_ylim(min_Y-.05,max_Y+.05)
    ax[1,1].set_xlim(min_X-.05,max_X+.05)
    ax[1,1].set_xlabel(r'$X$')
    ax[1,1].set_ylabel(r'$Y$')
    ax[1,1].legend(loc='best')

    fig.tight_layout()
    plt.savefig("ConSyn_gt_plots/example_plot/ConSyn_example.pdf")


if 1:
    # Plot now the reconstruction for the confounder model

    with open('config_makeConSyn.json', 'r') as config:
        setp = json.load(config)

    reconst_input = "/home/joka/Documents/Andrija_Faks/WiSe_19_20/IFT/BCI/scp_prelude/tmp_freya_ConSyn/pair0087/"

    direction = "X<-Z->Y"
    version = "v4"

    cm = Causal_Model(direction, data=[X, Y], config=setp, version=version)
    model = select_model(cm)

    N_samples = 2; N_steps = 60

    path = reconst_input
    path += "/N_samples_{}_N_steps_{}/".format(N_samples, N_steps) 

    seed = model.model_dict['seed']
    f_ID = path + direction + "_" + "KL_position_version_{}_{}.npy".format(version, seed) 
    KL_position = load_KL_position(f_ID)

    KL_position = ift.makeField(model._Ham.domain, KL_position)

    f_ID = path + "samples/" + direction + "*version_{}*_{}_*".format(version, seed)
    samples = glob.glob(f_ID)

    positions = []
    for file in samples:
        sample = load_KL_sample(file)
        sample = ift.makeField(\
                    model._Ham.domain, sample)    
        positions.append(KL_position + sample)

    f_X_list = []
    f_X_list_unsorted = []
    f_Y_list = []
    f_Y_list_unsorted = []
    U_list = []
    full_X = []
    full_Y = []
    sigma_inv_X_list = []
    sigma_inv_Y_list = []
    f_X_ps_list = []
    f_Y_ps_list = []

    sc_u = ift.StatCalculator()
    sc_f_X = ift.StatCalculator()
    sc_f_Y = ift.StatCalculator()

    for pos in positions:

        # Put the output fields in right order of indices
        # w.r.t. to the z-field

        u = model._U.force(pos).val
        idx = u.argsort()

        U = ift.makeField(model._U.target, u[idx])
        sc_u.add(U)

        U_list.append(u[idx])

        f_X_op = (model._f_X_op).force(pos)
        f_X_list_unsorted.append(f_X_op)
        f_X_op_sorted = ift.makeField(f_X_op.domain,f_X_op.val[idx])
        f_X_list.append(f_X_op_sorted)

        sc_f_X.add(f_X_op_sorted)

        f_Y_op = (model._f_Y_op).force(pos)
        f_Y_list_unsorted.append(f_Y_op)
        f_Y_op_sorted = ift.makeField(f_Y_op.domain, f_Y_op.val[idx])
        f_Y_list.append(f_Y_op_sorted)

        sc_f_Y.add(f_Y_op_sorted)

        f_X_ps_list.append(model._amp_f_x.force(pos))
        f_Y_ps_list.append(model._amp_f_y.force(pos))

        full_X.append(model._corr_f_x.force(pos))
        full_Y.append(model._corr_f_y.force(pos))

        sigma_inv_X_list.append((model._sigma_inv_X**(-1)).sqrt().force(pos))
        sigma_inv_Y_list.append((model._sigma_inv_Y**(-1)).sqrt().force(pos))


    scale = (0,1)
    X,Y = _normalize(X, Y, scale)

    min_Y, max_Y = min(Y), max(Y)
    min_X, max_X = min(X), max(X)

    fig, ax = plt.subplots(2,2, figsize=(10,10))

    U_mean = sc_u.mean.val
    f_X_mean = sc_f_X.mean.val
    f_Y_mean = sc_f_Y.mean.val

    hist_u, edges_u = np.histogram(U_mean, bins=pdf_Z.size)
    x_axis_u = 0.5*(edges_u[1:] + edges_u[:-1])

    x_pdf = np.linspace(0, 1, num=pdf_Z.size)
    ax[0,0].plot(x_pdf, hist_u[::-1], 'k-', alpha=.5, linewidth=.5, label='Hist. of the mean of sampled U values')
    ax[0,0].plot(x_pdf, pdf_Z/max(pdf_Z), 'k-', linewidth=2, alpha=1.0, label=r'$P_z$')
    ax[0,0].set_xlim(-.05,1.05)
    ax[0,0].set_xlabel(r'$z$')
    ax[0,0].set_ylabel('Counts')
    ax[0,0].legend(loc='best')

    leg = ax[0,0].legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    scaler = MinMaxScaler(scale)     
    
    f_X_scaled = scaler.fit_transform(f_X.reshape(-1,1)).T[0]
    f_Y_scaled = scaler.fit_transform(f_Y.reshape(-1,1)).T[0]
    
    count = 0
    for f_op, u in zip(f_X_list, U_list):
        if count == 0:
            ax[0,1].plot(f_op.val, u, 'b-', linewidth=.2, alpha=.2, label=r'Posterior samples  $f_X^{(i)}(U)$')
            count+=1
        else:
            ax[0,1].plot(f_op.val, u, 'b-', linewidth=.2, alpha=.2)

    ax[0,1].plot(f_X_mean, U_mean, 'b--', linewidth=2.0, alpha=1.0, label="Posterior mean")

    indx = Z.argsort()
    ax[0,1].plot(f_X_scaled[indx], Z[indx], 'k-', linewidth=2., label=r'$f_X(Z)$')
    ax[0,1].set_ylim(-.05,1.05)
    ax[0,1].set_xlim(min_X-0.05,max_X+0.05)
    ax[0,1].set_xlabel(r'$f_X(Z)$ and $f_X^{(i)}(U)$')
    ax[0,1].set_ylabel(r'$Z$ for $f_X(Z)$ and $U$ for $f_X^{(i)}(U)$' )
    ax[0,1].legend(loc='best')

    leg = ax[0,1].legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    count = 0
    for f_op, u in zip(f_Y_list, U_list):
        if count == 0:
            ax[1,0].plot(u, f_op.val, 'b-', linewidth=.2, alpha=.2, label=r'Posterior samples $f_Y^{(i)}(U)$')
            count+=1
        else:
            ax[1,0].plot(u, f_op.val, 'b-', linewidth=.2, alpha=.2)

    ax[1,0].plot(U_mean, f_Y_mean, 'b--', linewidth=2.0, alpha=1.0, label="Posterior mean")

    ax[1,0].plot(Z[indx], f_Y_scaled[indx], 'k-', linewidth=2., label=r'$f_Y(Z)$')
    ax[1,0].set_ylim(min_Y-0.05,max_Y+0.05)
    ax[1,0].set_xlim(-.05,1.05)
    ax[1,0].set_ylabel(r'$f_Y(Z)$ and $f_Y^{(i)}(U)$')
    ax[1,0].set_xlabel(r'$Z$ for $f_Y(Z)$ and $U$ for $f_Y^{(i)}(U)$' )
    ax[1,0].legend(loc='best')

    leg = ax[1,0].legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    ax[1,1].plot(X,Y, 'ko', alpha=.1, label="Observed data")
    ax[1,1].plot(f_X_scaled[indx],f_Y_scaled[indx], 'r-', linewidth=2., alpha=1.0, label="Ground truth")
    
    count = 0
    for f_op_X, f_op_Y in zip(f_X_list, f_Y_list):
        if count == 0:
            ax[1,1].plot(f_op_X.val, f_op_Y.val, 'b-', linewidth=.4, alpha=.2, label=r'Posterior samples $f_Y^{(i)}(U)$')
            count+=1
        else:
            ax[1,1].plot(f_op_X.val, f_op_Y.val, 'b-', linewidth=.4, alpha=.2)

    ax[1,1].plot(f_X_mean, f_Y_mean, 'b--', linewidth=2.0, alpha=1.0, label="Posterior mean")

    ax[1,1].set_ylim(min_Y-.05,max_Y+.05)
    ax[1,1].set_xlim(min_X-.05,max_X+.05)
    ax[1,1].set_xlabel(r'$X$')
    ax[1,1].set_ylabel(r'$Y$')
    ax[1,1].legend(loc='best')

    leg = ax[1,1].legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    
    fig.tight_layout()
    plt.savefig("ConSyn_gt_plots/example_plot/ConSyn_reconst_of_example.pdf")