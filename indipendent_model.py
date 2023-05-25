import numpy as np

import nifty6 as ift

from causal_model import Causal_Model
from plotting_utilities import myPlot
from model_utilities import guess_k_indx

class Indipendent_model(Causal_Model):

    def __init__(self, cm, verbose=False):

        super().__init__(cm.direction, [cm.X, cm.Y], cm.config)

        domain = self.domain

        # I would need extended domain
        if isinstance(domain, ift.RGSpace):
            # Working with 1D case for the moment
            extended_domain = ift.RGSpace((2*self.nbins),
                                          distances=1./self.nbins)
        else:
            raise NotImplementedError

        Hams = []
        R_list = []
        corr_beta_list = []
        amp_beta_list = []
        ln_likelihood_list = []
        Counts_fld_list = []

        # By default take the setup of the real model for inference
        model = self.config['real_model'][self.direction]
        ps_key = 'beta'

        keys = ["beta_X", "beta_Y"]
        for data, key in zip([X, Y], keys):
            Counts_fld, _ = self._setup_cause_effect_flds(data, np.zeros(0))

            ln_likelihood_beta, R_lamb, corr_fld_beta, amp_beta = \
                self.lognormal_model_setup(
                    model, ps_key, extended_domain, Counts_fld, name=key + '_')

            ln_likelihood_list.append(ln_likelihood_beta)
            R_list.append(R_lamb)
            corr_beta_list.append(corr_fld_beta)
            amp_beta_list.append(amp_beta)
            Counts_fld_list.append(Counts_fld)

        # Setup te model for Y

        ln_likelihood_tot = np.sum(ln_likelihood_list)

        # FIXME: Currently this is an overkill, but signifies
        # that the goal is to go towards a more complicated graph
        # structure, with >2 nodes

        self._Ham = self._initialize_Hamiltonians([ln_likelihood_tot])[0]
        self.keys = keys

        self._counts_fld_X, self._counts_fld_Y = Counts_fld_list[0], Counts_fld_list[1]
        
        self._beta_X, self._beta_Y = R_list[0], R_list[1]
        self._corr_beta_X, self._corr_beta_Y, = corr_beta_list[0], corr_beta_list[1]
        self._ps_X, self._ps_Y = amp_beta_list[0], amp_beta_list[1]

        # FIXME:
        # Think about what value to set here for X || Y
        #
        # POTENTIAL SOLUTION:
        # 
        # Assume here that the noise produced is poissonic
        # hence the noise level would be ~ 1/ sqrt(N), where
        # N is the number of counts in total, i.e. number of data
        # points

        self._sigma_p_X = \
            ift.ScalingOperator(self._ps_X.target, np.sqrt(X.size))
        self._sigma_p_Y = \
            ift.ScalingOperator(self._ps_Y.target, np.sqrt(Y.size))
    
        self.point_estimates=[]
        self._initial_mean = 0.1 * \
            ift.from_random(self._Ham.domain, "normal")

        if verbose:

            from playground import playground_indip

            ops = {}
            ops['beta_X'], ops['beta_Y'] = self._beta_X, self._beta_Y;
            ops['corr_beta_X'], ops['corr_beta_Y'] = self._corr_beta_X, self._corr_beta_Y
            ops['ps_X'], ops['ps_Y'] = self._ps_X , self._ps_Y
            ops['X_fld'], ops['Y_fld'] = self.fld_X, self.fld_Y
            ops['Ham'] = self._Ham
            ops['counts_fld_X'] = self._counts_fld_X;
            ops['counts_fld_Y'] = self._counts_fld_Y;
            ops['minimizer'] = self.minimizer

            playground_indip(ops)
            exit()
    
    def _k_indx(self, positions):

        k_indx_X, k_indx_Y = \
            guess_k_indx(self._sigma_p_X, self._ps_X, positions, direction=self.direction), \
            guess_k_indx(self._sigma_p_Y, self._ps_Y, positions, direction=self.direction)

        return max(k_indx_X, k_indx_Y)

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            pos = 0.1*ift.from_random(self._Ham.domain, 'normal')
            positions.append(pos)

        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 2)
        ny = kwargs.pop('ny', 3)
        xsize = kwargs.pop('xsize', 16)
        ysize = kwargs.pop('ysize', 16)

        beta_X = []
        beta_Y = []
        full_beta_X = []
        full_beta_Y = []
        beta_X_ps = []
        beta_Y_ps = []

        for pos in positions:
            beta_X.append(self._beta_X.force(pos))
            beta_Y.append(self._beta_Y.force(pos))
            full_beta_X.append((self._corr_beta_X.exp()).force(pos))
            full_beta_Y.append((self._corr_beta_Y.exp()).force(pos))
            beta_X_ps.append(self._ps_X.force(pos))
            beta_Y_ps.append(self._ps_Y.force(pos))

        # Plot beta_X setup
        plot = myPlot()

        # Set up the xcoords for the Cnts fld

        xcoord_cnts = np.linspace(0.,1.0,self._counts_fld_X.size)

        shp = len(beta_X)
        xcoord = len(beta_X) * [xcoord_cnts] + [xcoord_cnts]
        marker = list(np.full(shp, None)) + ['x']
        alpha = list(np.full(shp, .8)) + [.3]
        labels = list(np.full(shp, "")) + ["Data"]
        scatter = list(np.full(shp, False)) + [True]
    

        plot.my_add(beta_X + [self._counts_fld_X], label=labels,\
                    xcoord=xcoord, scatter=scatter, marker=marker,\
                    alpha=alpha,\
                    title=r"$\beta_X$")
        
        plot.my_add(beta_Y + [self._counts_fld_Y], label=labels,\
                    xcoord=xcoord, scatter=scatter, marker=marker, \
                    alpha=alpha,
                    title=r"$\beta_Y$")
        
        plot.my_add(full_beta_X, \
            alpha=alpha,\
            xmin=-0.5,xmax=1.5,ymin=-0.05,ymax=max(self._counts_fld_X.val)+1,\
            title=r"$\beta_X$ full")

        plot.my_add(full_beta_Y, \
            alpha=alpha,\
            xmin=-0.5,xmax=1.5,ymin=-0.05,ymax=max(self._counts_fld_Y.val)+1,\
            title=r"$\beta_Y$ full")

        plot.my_add(beta_X_ps, title=r'$p(\beta_X)$')
        
        plot.my_add(beta_Y_ps, title=r'$p(\beta_Y)$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
                       name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)
