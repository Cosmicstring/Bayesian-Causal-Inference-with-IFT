import matplotlib.pyplot as plt
import nifty6 as ift
import numpy as np

from nifty6.sugar import power_analyze

ift.random.push_sseq_from_seed(42)

N = 512
position_space = ift.RGSpace(N,)


offset_amplitude_mean = 1.0
offset_amplitude_stddev = 0.5
offset_amplitude_stddev_stddev = 0.5

cfmaker = ift.CorrelatedFieldMaker.make(\
	offset_amplitude_mean, \
	offset_amplitude_stddev, \
	offset_amplitude_stddev_stddev,\
	'')
cfmaker.add_fluctuations(position_space,
                             1., 1e-2, 1, .5, .1, .5, -3, 0.5, '')
correlated_field = cfmaker.finalize()
Amplitude = cfmaker.amplitude

fluctuations_mean = 1e2
fluctuations_stddev = 1e1
flexibility_mean = 1e-1
flexibility_stddev = 1e-2
asperity_mean = 1e-1
asperity_stddev = 1e-2
loglogavgslope_mean = -4.0
loglogavgslope_stddev = 0.5

xi = ift.from_random(correlated_field.domain, 'normal')
correlated_field_realization = correlated_field(xi)

xi_spectrum = ift.from_random(Amplitude.domain, 'normal')

for i in range(10):
	xi = ift.from_random(correlated_field.domain, 'normal')
	xi_spectrum = ift.from_random(Amplitude.domain, 'normal')

	# offset_amplitude_mean *= 2
	# offset_amplitude_stddev *= 10
	fluctuations_mean *= 2
	
# QUESTION:	I don't understand why does the increase of fluctuations_stddev decrease
# the total power in the powerspectrum?

	# fluctuations_stddev *= 2

# Controls the amplitude of the deviations in the powerspectrum from beginning
# to the end

	# flexibility_mean *= 2

# Fixed endpoints and controls the deviation in the middle

	# flexibility_stddev *= 2 
	
# Controls the smale scale variation amplitude

	# asperity_mean *= 2 

# QUESTION: Why does the increase in the asperity_stddev bring black curve
# close to the red curve?

	# asperity_stddev /= 10

	# loglogavgslope_mean *= 2

# QUESTION: Why does a steeper power spectrum, i.e. when loglogavgslope is increased
# make the smal scale fluctuations more prominent?

	# loglogavgslope_stddev *= 2
	
	cfmaker = ift.CorrelatedFieldMaker.make(\
		offset_amplitude_mean, \
		offset_amplitude_stddev, \
		offset_amplitude_stddev_stddev, \
		'')
	cfmaker.add_fluctuations(position_space,
                             fluctuations_mean, 
                             fluctuations_stddev, 
                             flexibility_mean,
                             flexibility_stddev,
                             asperity_mean,
                             asperity_stddev,
                             loglogavgslope_mean,
                             loglogavgslope_stddev, '')

	_correlated_field = cfmaker.finalize()
	_correlated_field_realization = _correlated_field(xi)
	_Amplitude = cfmaker.amplitude

	fig, ax = plt.subplots(nrows = 2, ncols = 1)
	ax[0].plot(_correlated_field_realization.val, color='black')
	# ax[0].plot(correlated_field_realization.val, color='red')
	
	ax[1].plot(_Amplitude(xi_spectrum).val, color='black')
	# ax[1].plot(Amplitude(xi_spectrum).val, color = 'red')
	ax[1].set_xscale('log'); ax[1].set_yscale('log')
	plt.show()
