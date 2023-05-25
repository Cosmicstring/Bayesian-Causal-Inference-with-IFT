import nifty6 as ift
import numpy as np
import pytest
import random

import matplotlib.pyplot as plt

from itertools import product
from scipy.stats import norm

import sys
sys.path.append("/home/joka/Documents/Andrija_Faks/WiSe_19_20/IFT/BCI/andrija-kostic/Bayesian_Causal_Inference/implementation")

from operator_utilities import UniformSamples, CDF, CmfLinearInterpolator, myInterpolator, \
IntegrateField, NormalizeField, normalize, rescalemax, findmax, findmin, Confounder_merge

# import pdb; pdb.set_trace()

def field(request):
    with ift.random.Context(request[0]):
        S = ift.ScalingOperator(request[1], 1.)
        return S.draw_sample_with_dtype(np.float64)

def test_findmin(field):

    op = findmin(field)

def test_variablecovariancegaussian(field):
    dc = {'a':field, 'b': field.exp()}
    mf = ift.MultiField.from_dict(dc)
    energy = ift.VariableCovarianceGaussianEnergy(field.domain,
        'a', 'b')
    ift.extra.check_jacobian_consistency(energy, mf, tol=1e-7)

def test_Confounder_merge(field):

    tgt = ift.makeDomain(ift.RGSpace((2*field.size, )))

    op = Confounder_merge(field.domain, 'a', field.domain, 'b', tgt)

    ift.extra.consistency_check(op)

def test_CDF(field):
    
    exp_field = field.exp()
    
    norm = normalize(exp_field.domain)
    exp_field_norm = norm(exp_field)

    op = CDF(exp_field_norm.domain)
    rescale = rescalemax(op.target)

    op = rescale(op)

    ift.extra.check_jacobian_consistency(op, exp_field_norm, tol=1e-7)

def test_UniformSamples(field):
    op = UniformSamples(field.domain)
    ift.extra.check_jacobian_consistency(op, field, tol=1e-9)

def test_CSOperator(field):
    op = CSOperator(field.domain)
    ift.extra.check_jacobian_consistency(op, field, tol=1e-8)

def test_and_plot_Interpolator(field):
    # The ground truth
    f_x = lambda x: 0.5*np.sin(5.*x)

    # Points for which I want to deduce value of f_fld
    z_prim = []
    for _ in range(10):
        z_prim.append(random.uniform(0., 1.5*np.pi))
    
    z_prim = np.asarray(z_prim)

    f_true_at_z_prim = f_x(z_prim)

    # Sampled points
    N=512
    x = np.linspace(0., 2*np.pi, num=N, endpoint=True)

    # FIXME: I have to think how to set up limits for my z
    # domain. Maybe I can use the rescaling to [0,1] through
    # the 

    x = np.asarray(x)
    f_at_x = f_x(x)

    # When making the domain for f_fld I have to take into
    # account the range on which my z_prim field can live on.
    # Here it is [0,2*np.pi]
    dom_f = ift.makeDomain(\
        ift.RGSpace(N, distances=2*np.pi/N))

    # Setup fields
    f_fld = ift.Field(\
        dom_f,\
        f_at_x)
    z_prim = ift.Field(\
        ift.makeDomain(ift.UnstructuredDomain(z_prim.size)),\
        z_prim)
    
    mf = {'f' : f_fld, 'z_prim' : z_prim}
    mf = ift.MultiField.from_dict(mf)
    op = myInterpolator(f_fld.domain, 'f', z_prim.domain, 'z_prim', shift=False,\
        min_z=0., max_z=2*np.pi)

    # Interpolated points
    interp = op(mf)

    plt.plot(x, f_at_x, 'b--')
    plt.plot(z_prim.val, f_x(z_prim.val), 'ro')
    plt.plot(z_prim.val, interp.val, 'gx')
    plt.show()

def test_Interpolator(field):

    # Take in the z_field from the CmfLinearInterpolator

    exp_field = field.exp()
    u = ift.Field(\
        ift.makeDomain(ift.UnstructuredDomain(field.domain.shape)), \
        field.val)

    UniS = UniformSamples(u.domain).ducktape('u')
    cdf = CDF(exp_field.domain).ducktape('cdf')

    rescale = rescalemax(cdf.target)

    # Need to normalize cdf before forwarding to
    # CmFLinearInterpolator
    cdf = rescale(cdf)

    input_op = \
    ift.FieldAdapter(cdf.target, 'cdf_key').adjoint @ cdf + \
    ift.FieldAdapter(UniS.target, 'u_key').adjoint @ UniS

    op = CmfLinearInterpolator(cdf.target, 'cdf_key', UniS.target, 'u_key', verbose=False)    
    op = op(input_op)

    _mf = ift.MultiField.from_dict({'cdf' : exp_field, 'u' : u})

    z_prim = op(_mf)

    # Imagine that the rg_dom is living on [-1,2] range 
    rg_dom = ift.makeDomain(\
        ift.RGSpace(field.shape, distances=3.0/field.size))
    z_dom = z_prim.domain
    
    #Change domain for field
    field = ift.Field(rg_dom, field.val)

    mf = {'f' : field, 'z_prim' : z_prim}
    mf = ift.MultiField.from_dict(mf)

    op = myInterpolator(rg_dom, 'f', z_dom, 'z_prim', verbose=False)

    ift.extra.check_jacobian_consistency(op, mf, tol=1e-7)


def test_CmfLinearInterpolator(field):

    exp_field = field.exp()
    u = ift.Field(\
        ift.makeDomain(ift.UnstructuredDomain(field.domain.shape)), \
        field.val)

    UniS = UniformSamples(u.domain).ducktape('u')
    cdf = CDF(exp_field.domain).ducktape('cdf')

    rescale = rescalemax(cdf.target)

    # Need to normalize cdf before forwarding to
    # CmFLinearInterpolator
    cdf = rescale(cdf)

    input_op = \
    ift.FieldAdapter(cdf.target, 'cdf_key').adjoint @ cdf + \
    ift.FieldAdapter(UniS.target, 'u_key').adjoint @ UniS

    op = CmfLinearInterpolator(cdf.target, 'cdf_key', UniS.target, 'u_key', verbose=False)    
    op = op(input_op)

    mf = ift.MultiField.from_dict({'cdf' : exp_field, 'u' : u})

    ift.extra.check_jacobian_consistency(op, mf, tol=1e-5)

def test_IntegrateField(field):

    op = IntegrateField(field.domain)
    ift.extra.check_jacobian_consistency(op, field, tol=1e-7)

def test_normalize(field):

    op = normalize(field.domain)
    ift.extra.check_jacobian_consistency(op, field, tol=1e-7)

def test_rescalemax(field):

    op = rescalemax(field.domain)
    ift.extra.check_jacobian_consistency(op, field)

if __name__ == "__main__":
    
    SPACES = [ift.RGSpace(512, distances=.789)]
    # for sp in SPACES[:3]:
    #     SPACES.append(ift.MultiDomain.make({'asdf': sp}))
    SEEDS = [4, 78, 23]
    PARAMS = product(SEEDS, SPACES)
        
    for request in PARAMS:
        print(request[0], request[1])
        test_Confounder_merge(field(request))