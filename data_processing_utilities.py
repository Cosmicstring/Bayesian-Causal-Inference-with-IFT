import nifty6 as ift
from nifty6 import makeDomain
from nifty6 import Field

from model_utilities import generate_analytic_ps, generate_parametric_ps, get_corr_and_amp

import numpy as np
import os

import argparse

neglected_files = \
        {
            "tcep_no_des" :
            {"pair0052.txt", "pair0053.txt", "pair0054.txt", "pair0055.txt",
             "pair0071.txt", "pair0105.txt"},
            "tcep_subsampled" : {},
            "ConSyn" : {},
            "SIM" : {},
            "SIM-c" : {},
            "SIM-G" : {},
            "SIM-ln" : {},
            "bcs_default": {},
            "synthetic" : {}
        }

class Parser(argparse.ArgumentParser):

    def __init__(self):
        super().__init__(description="Argument parser for BCI program")

        self.add_argument(
            "--N_samples", type=int, default=2, \
            help="Number of samples to start the KL minimization from.")

        self.add_argument(
            "--N_steps", type=int, default=2, \
            help="Number of steps for the global iterations of the KL minimization.")

        self.add_argument(
            "--analyse", type=int, choices=[0,1], default=0, \
            help="If set to '1' then just calculate the evidence of the corresponding models, otherwise do the full inference.")

        self.add_argument(
            "--config", type=str, default='config.json',\
            help="Select the configuration file which contains the information on hyperparameters for the models.")

        self.add_argument(
            "--version", type=str, default='v1', \
            help="Select version of the corresponding model. Take a look into `select_model.py`.")

        self.add_argument(
            "--benchmark", type=str, default='bcs_default', \
            help="Select the benchmark to test the code on. Take a look at `benchmark_tests` directory.")

        self.add_argument(
            "--direction", type=str, default='X->Y',\
            help="Select the causal direction for which you're interested in.")

        self.add_argument(
            "--batch", type=int, default=1, \
            help="This controls the testcase batch, useful if one wants to run on smaller chunks of the benchmark dataset.")

    def batches(self, args):

        tcep_batch = \
        {
            0 : slice(0,10),
            1 : slice(16,20),
            2 : slice(20,30),
            3 : slice(30,40),
            4 : slice(40,50),
            5 : slice(50,60),
            6 : slice(60,70),
            7 : slice(70,80),
            8 : slice(80,90),
            9 : slice(90,100),
            10 : slice(100,108)
        }

        bcs_default = \
        {
            1 : slice(0,10),
            2 : slice(10,20),
            3 : slice(20,30),
            4 : slice(30,40),
            5 : slice(40,50),
            6 : slice(50,60),
            7 : slice(60,70),
            8 : slice(70,80),
            9 : slice(80,90),
            10: slice(90,100)
        }


        SIM_c = \
        {
            0: slice(1,10),
            1: slice(10,15),
            2: slice(15,20),
            3: slice(20,30)
        }


        if args.benchmark == 'tcep_no_des' or \
           args.benchmark == 'tcep_no_des_units' or \
           args.benchmark == 'tcep_subsampled':

            return tcep_batch[args.batch]

        if args.benchmark == 'bcs_default' or \
           args.benchmark == 'SIM' or \
           args.benchmark == 'SIM-G' or \
           args.benchmark == 'SIM-ln' or \
           args.benchmark == 'ConSyn':
            return bcs_default[args.batch]

        if args.benchmark=='synthetic':
            return slice(0,1)

        if args.benchmark=='SIM-c':
            return SIM_c[args.batch]

def _save_data(X, Y, filename, path):

    output = os.path.join(path, filename)
    f = open(output, "w")
    for x, y in zip(X, Y):
        f.write("{:.18f}\t{:.18f}\n".format(x, y))
    f.close()


def _readin_data(filename, path):

    _input = os.path.join(path, filename)

    data = np.loadtxt(_input)
    # Note that we are only considering
    # 1D data from the dataset. One has to ensure
    # there are only 2 relevant columns of data in
    # the input file!
    return data[:,0],data[:,1]


def _normalize(X, Y, scale):

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(scale)
    X, Y = scaler.fit_transform(np.array((X, Y)).T).T

    return X, Y


def generate_indip_data():
    pass


def generate_bivar_data(model_setup):

    seed = model_setup["seed"]
    ift.random.push_sseq_from_seed(seed)

    position_space_shape = model_setup["shape_spaces"]["position_space"]

    position_space = ift.RGSpace(position_space_shape)

    amp_beta_op, correlated_field_beta = \
    get_corr_and_amp(
        model_setup,
        model_setup["beta"]["ps_flag"],
        "beta",
        position_space,
        "beta")

    lamb = ift.exp(correlated_field_beta)

    # Response will here be simulating the detector
    # which masks certain regions of the x_space

    R = ift.GeometryRemover(position_space)
    detector_mask = np.random.binomial(1, 0.1, position_space.shape)
    detector_mask = ift.Field(R.target, detector_mask)
    Mask = ift.MaskOperator(detector_mask)

    # The masked regions are the ones where detector_mask == 1, and the
    # unmasked ones are with detector_mask == 0
    Reference_x = ift.MaskOperator(1 - detector_mask)

    train_lamb = Mask(R(lamb))

    # And ground_truth rate
    lambda_ = R(lamb)

    k_space = train_lamb.target

    # White gaussian sample to get a realization for my lambda_ and trian_lamb fields
    xi = ift.from_random(lamb.domain, 'normal')

    true_exp_beta_x_data = lambda_(xi)
    train_exp_beta_x_data = train_lamb(xi)

    true_k_data = np.random.poisson(true_exp_beta_x_data.val.astype(np.float64))
    train_k_data = np.random.poisson(train_exp_beta_x_data.val.astype(np.float64))

    true_k_data_fld = ift.Field.from_raw(lambda_.target, true_k_data)
    train_k_data_fld = ift.Field.from_raw(k_space, train_k_data)

    # Now I need to build up my x_data according to the above realization
    # for my mock y_data
    #
    # NOTE: Here I first need to build up a new position space on which
    # the x_field will be stored, since k_fld and lambda_.target are UnstructuredDomain,
    # and is 1D for now

    npoints = np.sum(true_k_data)
    temp_position_space = ift.RGSpace(true_k_data.shape)

    dom = temp_position_space
    dist = temp_position_space.distances[0]

    # In NIFTY RGSpace ranges from 0 to 1!
    dist_arr = np.arange(0, 1, dist)

    true_x_data = np.empty((npoints,))
    indx = 0

    # Here the sufficient number of x_data within the given bins
    # is generated
    for i in range(len(dist_arr)-1):
        k = true_k_data[i]
        left = dist_arr[i]; right = dist_arr[i+1]
        for _ in range(k):
            x_val = np.random.uniform(left, right)
            true_x_data[indx] = x_val
            indx += 1

    X = true_x_data

    x_position_space = ift.RGSpace(X.shape)

    true_x_data_fld = ift.Field(makeDomain(x_position_space), X)

    # Now one has to interpolate to the values where true_x_data_fld
    # is defined

    interpolator = ift.LinearInterpolator(
        x_position_space, true_x_data_fld.val.reshape(1, -1))

    amp_f_op, correlated_field_f = \
    get_corr_and_amp(
        model_setup,
        model_setup['f']["ps_flag"],
        'f',
        x_position_space,
        'f')

    f_xi = ift.from_random(correlated_field_f.domain, 'normal')
    f_true = correlated_field_f(f_xi)

    # We use then these values for the minimization procedure
    f_at_x = interpolator(correlated_field_f)

    # For generating mock data, one uses the interpolated f_true
    f_true_at_x = interpolator(f_true)

    # Now the y_data would be built out of this interpolated f_true
    y_data_space = f_at_x.target

    # Noise would be just scalar noise
    noise = 0.01
    N = ift.ScalingOperator(y_data_space, noise)

    true_y_data = f_true_at_x
    train_y_data = f_true_at_x + N.draw_sample()

    Y = train_y_data.val

    return X, Y


def generate_confounder_data(file, seed, model_setup):

    import matplotlib.pyplot as plt
    import random

    from model_utilities import Confounder_model

    shp = model_setup['shape_spaces']['position_space']

    # FIXME: Implementation is done in such a way that you need to forward the
    # data anyways in order to get the model operators out, which is very strange.
    # Fix this.

    cm = Causal_Model(direction, data=[np.ones(shp), np.ones(shp)], version='v1', config=setup)
    model = cm.select_model()

    xi = {}
    dom = model._Ham.domain
    for key in dom.keys():
            if not (key in model.op_icdf.domain.keys()) and key != 'u':
                xi[key] = ift.from_random(dom[key], 'normal')
            else:
                xi[key] = ift.from_random(dom[key], 'normal')

    xi = ift.MultiField.from_dict(xi)

    Z = model.op_icdf.force(xi).val
    f_X = model._f_X_op.force(xi).val
    f_Y = model._f_Y_op.force(xi).val

    plt.plot(Z, f_X, 'bx')
    plt.show()

    plt.plot(Z, f_Y, 'bx')
    plt.show()

    var_X = model._sigma_inv_X.force(xi).val**(-1)
    var_Y = model._sigma_inv_Y.force(xi).val**(-1)

    N_X = ift.ScalingOperator(ift.makeDomain(ift.UnstructuredDomain(shp)), var_X[0])
    N_Y = ift.ScalingOperator(ift.makeDomain(ift.UnstructuredDomain(shp)), var_Y[0])

    data_X = f_X + N_X.draw_sample_with_dtype(np.float64).val
    data_Y = f_Y + N_Y.draw_sample_with_dtype(np.float64).val

    # Output all the values into a .txt file for later use
    file.write(
        "{:15s}\t{:15s}\t{:15s}\t{:15s}\t{:15s}\n".format(
        "Z field", "f_X field", "data_X",
        "f_Y field", "data_Y"))

    for idx in range(Z.size):
        file.write(
            "{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\t{:.10f}\n".format(
            Z[idx],
            f_X[idx], data_X[idx],
            f_Y[idx], data_Y[idx]))

    return data_X, data_Y


def get_data(setup, filename=None, path=None):

    generate_mock_flag = setup["mock"]

    real_flag = setup["real"]
    data = []

    if generate_mock_flag == 1:
        # Use this default scale:
        scale = (0, 1)

        mock_setup = setup["mock_model"]
        causality_setup = mock_setup["model_flag"]

        if causality_setup == "X->Y":

            filename = "Mock_data_{}_.txt"

            X, Y = generate_bivar_data(mock_setup["X->Y"])
            X, Y = _normalize(X, Y, scale)

            filename = filename.format("X->Y")
            path = "mock_data/"
            _save_data(X, Y, filename, path)

        elif causality_setup == "X<-Z->Y":

            seed = mock_setup[causality_setup]["seed"]
            # File for output
            filename = "mock_data/Mock_data_" + \
                causality_setup + "_seed_" + "{:d}" + ".txt"

            f = open(filename.format(seed), "w")

            X, Y = generate_confounder_data(f, seed, mock_setup["X<-Z->Y"])

            f.close()

            X, Y = _normalize(X, Y, scale)

            path = "mock_data/"
            _save_data(X, Y, "Mock_data_X<-Z->Y.txt", path)

    elif real_flag == 1:

        X, Y = _readin_data(filename,path)

        scale = (0,1)
        X, Y = _normalize(X, Y, scale)

    else:
        raise ValueError("Not Implemented")

    if not isinstance(X, np.ndarray):
        X = np.asarray(X, dtype=np.float64)

    if not isinstance(Y, np.ndarray):
        Y = np.asarray(Y, dtype=np.float64)

    data.append(X)
    data.append(Y)

    return data


if __name__ == "__main__":

    import json
    from model_utilities import Causal_Model

    curr_path = os.path.abspath('.')
    if 'home/joka/' in curr_path:
        file_setup = open("config_laptop.json", "r")
    elif 'afs/mpa' in curr_path:
        file_setup = open("config.json", "r")

    setup = json.load(file_setup)
    direction = "X<-Z->Y"

    causality_setup = setup["mock_model"][direction]
    seed = 42

    filename = "mock_data/Mock_data_" + direction + "_seed_" + "{:d}" + ".txt"

    f = open(filename.format(seed), "w")

    X,Y = generate_confounder_data(f, seed, causality_setup)

    f.close()

    scale = (0,1)
    X,Y = _normalize(X, Y, scale)

    path = "mock_data/"
    _save_data(X, Y, "Mock_data_X<-Z->Y.txt", path)
