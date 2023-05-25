import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import CubicSpline

x_more = np.linspace(-2,2, 100)

# test the curve shape
for n in np.arange(5,20,3):
	x = np.linspace(-2,2,n)
	print(n)
	plt.scatter(x, norm.cdf(x))
	pchip_interp = PchipInterpolator(x, norm.cdf(x))
	cubic = CubicSpline(x, norm.cdf(x))
	
	plt.plot(x_more, norm.cdf(x_more), 'b-')
	plt.plot(x_more, pchip_interp(x_more), 'g--', label = 'pchip')
	plt.plot(x_more, cubic(x_more), 'r--', label='cubic')
	plt.legend()
	plt.show()

	plt.scatter(x, norm.pdf(x))
	plt.plot(x_more, norm.pdf(x_more), 'b-')
	plt.plot(x_more, pchip_interp.derivative()(x_more), 'g--', label='pchip')
	plt.plot(x_more, cubic.derivative()(x_more), 'r--', label='cubic')
	plt.legend()
	plt.show()


# Test the difference

for n in np.arange(10,100,10):
	x = np.linspace(-2,2,n)
	print(n)
	plt.scatter(x, np.zeros(x.size))
	pchip_interp = PchipInterpolator(x, norm.cdf(x))
	cubic = CubicSpline(x, norm.cdf(x))
	
	diff = (norm.cdf(x_more) - pchip_interp(x_more))

	plt.ylim(min(diff), max(diff))
	
	plt.plot(x_more, diff, 'g--', label = 'pchip')
	
	diff = (norm.cdf(x_more) - cubic(x_more))
	
	plt.plot(x_more, diff, 'r--', label='cubic')
	plt.show()

	plt.scatter(x, np.zeros(x.size))
	diff = (norm.pdf(x_more) - pchip_interp.derivative()(x_more))
	ymin = min(diff); ymax = max(diff)
	plt.plot(x_more, diff, 'g--', label='pchip')
	diff = (norm.pdf(x_more) - cubic.derivative()(x_more))
	plt.plot(x_more, diff, 'r--', label='cubic')
	plt.show()
