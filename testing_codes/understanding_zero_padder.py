import nifty6 as ift
import numpy as np
import data_processing_utilities as dpu
import matplotlib.pyplot as plt
import os
import json

from nifty6 import makeDomain

# import pdb; pdb.set_trace()

np.random.seed(111)

curr_path = os.path.abspath('.')
if 'home/joka/' in curr_path:
	file_setup = open("config_laptop.json", "r")
elif 'afs/mpa' in curr_path:
	file_setup = open("config.json", "r")

setup = json.load(file_setup)
file_setup.close()

X,Y = dpu.get_data(setup)

x_position_space = ift.RGSpace((3*X.size,))
x_harmonic_space = x_position_space.get_default_codomain()

f_HT = ift.HarmonicTransformOperator(x_harmonic_space,x_position_space)

# Define f_op powerspectra on this domain
# Get the spectra realization
cfmaker = ift.CorrelatedFieldMaker.make(1e-3, 1e-6, '')
cfmaker.add_fluctuations(x_position_space,
                             1., 1e-2, 1, .5, .1, .5, -3, 0.5, '')
correlated_field_f = cfmaker.finalize()
A_f = cfmaker.amplitude

# Make flags so that the central part stays free
flag_arr = np.ones(x_position_space.size)
lower_bound = int(flag_arr.size/3)
upper_bound = 2*lower_bound 
flag_arr[lower_bound:upper_bound] = 0	

flags = ift.Field(correlated_field_f.target, flag_arr)
mask = ift.MaskOperator(flags)

# To get a realization of my f_op field I apply it to white gaussian

f_at_x = mask(correlated_field_f)

f_xi = ift.from_random('uniform', f_at_x.domain)

# Make the domain with distances from original f_at_x domain
# which is padded

interpolator_domain_100 = ift.RGSpace(f_at_x.target.shape, \
	distances = 1./(f_at_x.target.size-100))#x_position_space.distances[0])
interpolator_domain_1  = ift.RGSpace(f_at_x.target.shape, \
	distances = 1./(f_at_x.target.size-1))

interpolator_100 = ift.LinearInterpolator(interpolator_domain_100, X.reshape(1,-1))
interpolator_1 = ift.LinearInterpolator(interpolator_domain_1, X.reshape(1,-1))

x_arr = np.arange(0,1,1./f_at_x.target.size)

plt.scatter(X, Y)
plt.plot(x_arr,f_at_x(f_xi).val)
# plt.show()
# exit()

# For switching between domains
GR_100 = ift.GeometryRemover(interpolator_domain_100)
GR_1 = ift.GeometryRemover(interpolator_domain_1)

# plt.plot(x_arr,GR.adjoint(f_at_x(f_xi)).val)

f_at_x_100 = interpolator_100(GR_100.adjoint(f_at_x))
f_at_x_1 = interpolator_1(GR_1.adjoint(f_at_x))

plt.scatter(X, f_at_x_100(f_xi).val, color='r', marker = 'x')
plt.scatter(X, f_at_x_1(f_xi).val, color='y', marker='x')
plt.show()
# plt.scatter(Y,X, marker='x')