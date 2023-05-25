import numpy as np
import os
import pickle
import nifty6 as ift

from nifty6 import EnergyOperator, VariableCovarianceGaussianEnergy
from operator_utilities import rescalemax, normalize, CDF, UniformSamples, \
    CmfLinearInterpolator, myInterpolator, GeomMaskOperator, Confounder_merge

from plotting_utilities import myPlot

class SingleDomain(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val)

def save_random_state(rstate_f):
    with open(rstate_f, 'wb') as f:
        pickle.dump(ift.random.getState(), f)

def load_random_state(rstate_f):
    with open(rstate_f, 'rb') as f:
        ift.random.setState(pickle.load(f))

def save_KL_position(KL_position, f_ID):
    np.save(f_ID, KL_position)

def save_KL_sample(KL_sample, f_ID):
    np.save(f_ID, KL_sample)

def load_KL_sample(f_ID):
    inp = np.load(file=f_ID, allow_pickle=True)
    return inp[()]

def load_KL_position(f_ID):
    inp = np.load(file=f_ID, allow_pickle=True)
    return inp[()]

def generate_analytic_ps(a, deg, offset, HT, harmonic_space):

    def sqrt_ps(k): return (a / (k**deg + offset))

    p_space = ift.PowerSpace(harmonic_space)
    pd = ift.PowerDistributor(harmonic_space, p_space)
    a = ift.PS_field(p_space, sqrt_ps)
    A = pd(a)

    A_op = ift.makeOp(A)
    A_op_pos_space = HT(A_op)

    return A_op, A_op_pos_space


def generate_parametric_ps(
    position_space,\
    offset_amplitude_mean,
    offset_amplitude_stddev,
    offset_amplitude_stddev_stddev,
    fluctuations_mean,
    fluctuations_stddev,
    flexibility_mean,
    flexibility_stddev,
    asperity_mean,
    asperity_stddev,
    loglogavgslope_mean,
    loglogavgslope_stddev,
    name='',):

    cfmaker = ift.CorrelatedFieldMaker.make(
        offset_amplitude_mean,
        offset_amplitude_stddev, offset_amplitude_stddev_stddev, name)
    cfmaker.add_fluctuations(position_space,
                             fluctuations_mean,
                             fluctuations_stddev,
                             flexibility_mean,
                             flexibility_stddev,
                             asperity_mean,
                             asperity_stddev,
                             loglogavgslope_mean,
                             loglogavgslope_stddev)
    correlated_field = cfmaker.finalize()
    A = cfmaker.amplitude

    return A, correlated_field


def get_corr_and_amp(\
    model, ps_flag, ps_key,
    domain,
    name):
        
    amp, correlated_field = generate_parametric_ps(\
                                            domain, \
                                            **model[ps_key]['ps_fluctuations'],\
                                            name=name)
    return amp, correlated_field


def Bin(grid_space, data):
    """
    grid_space : ift.RGSpace
            Space where the lognormal field is defined and
            at the same time defines the space where the binning
            is going to be done

    data :
            Data to be sampled at the bin positions as defined by the grid_space

    """

    # Get the distances array in order to be able to assign
    # the bins to the closes grid points of the RGSpace
    dist = grid_space.distances[0]
    nbins = grid_space.shape[0]

    bin_coords = np.arange(nbins, dtype=np.float64)*dist

    binned_data = np.zeros(nbins, dtype=np.int64)

    # Sort the data so that it is easier to count
    if isinstance(data, list):
        tmp = np.zeros(len(data))
        for i in range(len(data)):
            tmp[i] = data[i]
    elif isinstance(data, np.ndarray):
        tmp = data.copy()

    tmp.sort()

    # Copy into tmp array the data contents:
    indx = 0
    track = 0
    over_last = 0

    for x in tmp:

        # Remember here that the shape of the domain is
        # topologically a torus, therefore the points have
        # to wind up after one reaches the rightmost bin edge!

        if x > bin_coords[-1]:
            over_last += 1
        else:
            if x <= bin_coords[indx+1]:
                binned_data[indx] += 1
            else:
                while x > bin_coords[indx+1]:
                    indx += 1
                binned_data[indx] += 1

        track += 1

    binned_data[-1] += over_last

    return binned_data


def get_stats_and_plot(plot, KL, op,
                       data=None, ps=None,
                       xcoord_list=[None], scatter_list=[None],
                       color_list=[None],
                       alpha_list=[None],
                       xmin=None, xmax=None, ymin=None, ymax=None,
                       title=""):

    sc = ift.StatCalculator()

    for sample in KL.samples:
        sc.add(op.force(sample + KL.position))
     
    delta = ift.sqrt(sc.var)
    upper = sc.mean + delta
    lower = sc.mean - delta
    
    if isinstance(data, type(None)):
        plot.my_add([sc.mean, upper, lower],
                    scatter=scatter_list[1:],
                    xcoord=xcoord_list[1:],
                    color=color_list[1:],
                    alpha=alpha_list[1:],
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    title=title,
                    label=["mean", "upper limit", "lower limit"])
    else:
        plot_list = [data, sc.mean, upper, lower]
        label = ["data", "mean", "upper limit", "lower limit"]
        plot.my_add(plot_list,
                    scatter=scatter_list,
                    xcoord=xcoord_list,
                    color=color_list,
                    alpha=alpha_list,
                    xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                    title=title, label=label)

    if not isinstance(ps, type(None)):
        powers = [ps.force(s + KL.position) for s in KL.samples]

        # NOTE: Addition of two lists with nifty6.Field objects inside, actually just appends to the
        # list A the content of list B (assuming addition is done as A + B)

        plot.my_add(
            powers
            +
            [ps.force(KL.position)],
            title="Sampled Power Spectrum",
            # label=["posterior ps","true ps"],
            yscale='log',
            linewidth=[1.]*len(powers) + [3., 3.])

def plot_reconst(
        KL,
        op_list, ps_list,
        data_list, idx_list,
        coord_list,
        filename,
        keys,
        title="",
        xmin=None, xmax=None,
        ymin=None, ymax=None):

    for op, ps, data, key, idx in zip(op_list, ps_list, data_list, keys, idx_list):

        filename_res = filename.format(idx)

        # First one would be of the data, and the next three are for the posterior
        # reconst. field. There are three of them, since I am plotting upper, lower and
        # sc.mean of the reconstruction.

        xcoord_list = [coord_list['data_' + key],
                       coord_list[key], coord_list[key], coord_list[key]]

        tmp_scatter_list = [True, False, False, False]
        tmp_color_list = [None, 'black', 'red', 'orange']
        tmp_alpha_list = [0.5, None, None, None]

        plot = myPlot()
        get_stats_and_plot(plot, KL, op,
                           data=data,
                           ps=ps,
                           xcoord_list=xcoord_list, scatter_list=tmp_scatter_list,
                           color_list=tmp_color_list,
                           alpha_list=tmp_alpha_list,
                           xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
                           title=title)

        plot.my_output(nx=2, ny=1, xsize=16, ysize=6,
                       name=filename_res)
        print("Saved results as '{}'.".format(filename_res))

def guess_k_indx(sigma_inv, amp_f, positions, direction='X->Y', version='v1'):

    plot = myPlot()

    sc_amp_f = ift.StatCalculator()
    sc_sigma_ps = ift.StatCalculator()
    for xi in positions:
        sc_amp_f.add((amp_f.force(xi))**2)
        # Divide by the volume factor
        if direction != "X || Y" and (direction == "X<-Z->Y" and version != 'v5'):
            sc_sigma_ps.add((sigma_inv.force(xi))**(-1) / (amp_f.domain.size))
        else:
            # In this case noise is not directly
            # in the 'Ham' domain hence one needs
            # to draw another set of rndm numbers
            # for the realization of poissonic noise
            # in the independent model
            xi_p = ift.from_random(sigma_inv.domain, "normal")

            # FIXME: The absolute value added is for convenience,
            # but should be again thought through!

            sc_sigma_ps.add(np.abs((sigma_inv.force(xi_p))) / (amp_f.domain.size))

    sigma_mean = sc_sigma_ps.mean
    amp_f_mean = sc_amp_f.mean

    # We have constant sigma over the whole powerspace domain
    sigma_ps = ift.Field.full(amp_f.target, sigma_mean.val[0])

    plot.my_add([sigma_ps, amp_f_mean], scatter=[False, False], label=[r'$\sigma_n^2$', r'$A_f$'],
                title='Noise and signal prior powerspec')
    plot.my_output(ny=1, nx=1, name='test_{}.pdf'.format(direction))

    # Find the closest points
    k_min = np.inf
    k_indx = 0
    idx = 0
    fudge_factor = 10
    noise_scale = sigma_mean.val[0]
    for amp in amp_f_mean.val:
        delta = abs(amp - noise_scale)
        if delta < k_min:
            k_min = delta
            k_indx = idx + fudge_factor
        idx += 1

    return k_indx
