import numpy as np

import nifty6 as ift

from causal_model import Causal_Model
from plotting_utilities import myPlot
from operator_utilities import myInterpolator, GeomMaskOperator, normalize, rescalemax, CDF, CmfLinearInterpolator
from model_utilities import Bin, guess_k_indx, get_corr_and_amp
from minimization import stages

class Bipartite_model(Causal_Model):

    def __init__(self, cm, verbose=False):
        """
        Note here that the 'cm' is the instance of the parent class 'Causal_Model'
        which is forwaded here in order to be able to access its variables
        """
        super().__init__(cm.direction, [cm.X, cm.Y], cm.config, version=cm.version)

        domain = self.domain
        if self.direction=="X->Y":
            X, Y = cm.X, cm.Y
        elif self.direction=="Y->X":
            Y, X = cm.X, cm.Y
        else:
            raise NotImplementedError

        # By default take the setup of the real model for inference
        model = self.model_dict

        if not isinstance(domain, ift.Domain):
            raise ValueError("self.domain not an ift.Domain")

        # I would need extended domain
        if isinstance(domain, ift.RGSpace):
            # Working with 1D case for the moment
            self._extended_domain = ift.RGSpace((2*X.size), distances=1./X.size)
            self._extended_domain_beta = ift.RGSpace((2*self.nbins,),
                                               distances=1./self.nbins)
        else:
            raise NotImplementedError

        self.cause_fld, self.effect_fld = self._setup_cause_effect_flds(X, Y)
        self._X, self._Y = X, Y

        if self.version in ['v1','v2','v3','v4']:
            params = model[self.version]
        else:
            raise NotImplementedError

        # The models decouple for beta lognormal field and for the nonlinear response

        self._beta_ln_likelihood, \
            self._beta_op, self._corr_field_beta, \
            self._beta_ps = self.lognormal_model_setup(\
                params, 'beta',
                self._extended_domain_beta, self.cause_fld)

        if self.infer_noise == 0:
            self._f_op, self._f_ps = \
            self.nonlinresponse_model_setup(\
                params, 'f',
                self._extended_domain, X, self.effect_fld, self.infer_noise)

            self._f_ln_likelihood = (ift.GaussianEnergy(mean=effect_fld, inverse_covariance=N.inverse)
                               @ f_op)

        elif self.infer_noise == 1:
            self._f_op, self._f_ps, self._corr_field_f, \
            self._sigma_inv\
            = \
            self.nonlinresponse_model_setup(params, 'f',
                                          self._extended_domain, X, self.effect_fld, self.infer_noise)

            add_data = ift.Adder(-self.effect_fld)

            residual = add_data(self._f_op)

            # The likelihood would be the VariableCovGE since we're also inferring
            # the noise_cov (here just single sigma) which has appropriate metric for
            # this case
            FA_res = ift.FieldAdapter(residual.target, 'residual')
            FA_icov = ift.FieldAdapter(self._sigma_inv.target, 'icov')

            residual_at_icov = FA_res.adjoint @ residual + FA_icov.adjoint @ self._sigma_inv

            self._f_ln_likelihood = (ift.VariableCovarianceGaussianEnergy(\
                    self.effect_fld.domain,
                    'residual',
                    'icov',
                    np.float64) @ residual_at_icov)

        self._Ham = \
            self._initialize_Hamiltonians(
                [self._beta_ln_likelihood + self._f_ln_likelihood])[0]

        # FIXME: Think what is best here, whether to save all necesary variables from parent class
        # here locally, or refer to them with super() command, which would require that the class is
        # initialized here with 'super().__init__(*args)''
    
        self.point_estimates = []
        self._initial_mean = 0.1*ift.from_random(self._Ham.domain, 'normal')

        if verbose:

            from playground import playgrond_bipartite

            ops = {}
            ops['Ham'] = self._Ham
            ops['f_op'], ops['f_ps'] = self._f_op, self._f_ps
            ops['sigma_inv'] = self._sigma_inv
            ops['beta_op'] = self._beta_op
            ops['beta_ps'] = self._beta_ps
            ops['cause_fld'], ops['effect_fld'] = self.cause_fld, self.effect_fld
            ops['X'], ops['Y'] = self._X, self._Y
            ops['minimizer'] = self.minimizer

            playgrond_bipartite(ops)
            exit()

    @property
    def extended_domain(self):
        return self._extended_domain
    
    def _k_indx(self, positions):

        return guess_k_indx(self._sigma_inv, self._f_ps, positions, self.direction)

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            pos = 0.1*ift.from_random(self._Ham.domain, 'normal')
            positions.append(pos)

        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 2)
        ny = kwargs.pop('ny', 4)
        xsize = kwargs.pop('xsize', 15)
        ysize = kwargs.pop('ysize', 20)

        # Temporarily fix the seed
    
        f_op = []
        f_ps = []
        beta_op = []
        beta_ps = []
        full_f = []
        full_beta = []
        sigma_inv = []

        rg = ift.RGSpace(self._sigma_inv.target.shape)
        GR = ift.GeometryRemover(rg)

        for pos in positions:
            f_op.append(self._f_op.force(pos))
            f_ps.append(self._f_ps.force(pos))
            full_f.append(self._corr_field_f.force(pos))
            full_beta.append((self._corr_field_beta.exp()).force(pos))
            beta_op.append(self._beta_op.force(pos))
            beta_ps.append(self._beta_ps.force(pos))
            sigma_inv.append(\
                (GR.adjoint @ (self._sigma_inv**(-1)).sqrt()).force(pos))

        # Plot the setup
        plot = myPlot()

        shp = len(f_op)
        markers = list(np.full(shp, None)) + ['o']
        alpha = list(np.full(shp, .8)) + [.3]
        labels = list(np.full(shp, "")) + ["Data"]
        scatters = list(np.full(shp, False)) + [True]

        xcoords = list(np.outer(np.ones(shp), self._X)) + [self._X]

        plot.my_add(f_op + [self.effect_fld], label=labels,
                    xcoord=xcoords,\
                    marker=markers,  scatter=scatters, alpha=alpha,\
                    title=r'$\rm{f}$ samples')

        # Set up the xcoords for the Cnts fld

        dom = ift.RGSpace(self.cause_fld.domain.shape)
        dist = dom.distances[0]
        shp = dom.shape[0]

        xcoord_cnts = np.arange(shp, dtype=np.float64)*dist
        
        shp = len(beta_op)
        markers = list(np.full(shp, None)) + ['x']
        alpha = list(np.full(shp, .8)) + [.3]
        labels = list(np.full(shp, "")) + ["Data"]
        scatters = list(np.full(shp, False)) + [True]

        xcoords = list(np.outer(np.ones(shp), xcoord_cnts)) + [xcoord_cnts]

        plot.my_add(beta_op+[self.cause_fld], label=labels,
                    xcoord=xcoords, \
                    scatter=scatters, marker=markers, alpha = alpha,\
                    title=r'$\beta$ samples')

        plot.my_add(full_f, \
            xmin=-0.5,xmax=1.5,ymin=-0.05,ymax=1.5,\
            title=r"$\rm{f}$ full")

        plot.my_add(full_beta, \
            xmin=-0.5,xmax=1.5,ymin=-0.05,ymax=max(self.cause_fld.val)+1,\
            title=r"$\beta$ full")

        plot.my_add(sigma_inv, \
            title=r"$\sigma_f$")

        plot.my_add(f_ps, title=r'$p(\rm{f})$')

        plot.my_add(beta_ps, title=r'$p(\beta)$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
                       name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

class Bipartite_model_v2(Causal_Model):
    
    """
    Model with the structure U -> X -> Y 
    """

    def __init__(self, cm):

        super().__init__(cm.direction, [cm.X, cm.Y], cm.config,
                         version=cm.version)

        domain = self.domain
        if self.direction=="X->Y":
            X, Y = cm.X, cm.Y
        elif self.direction=="Y->X":
            Y, X = cm.X, cm.Y
        else:
            raise NotImplementedError

        # By default take the setup of the real model for inference
        model = self.model_dict

        if not isinstance(domain, ift.Domain):
            raise ValueError("self.domain not an ift.Domain")

        # I would need extended domain
        if isinstance(domain, ift.RGSpace):
            # Working with 1D case for the moment
            self._adapted_size_factor = 3
            self._extended_domain = ift.RGSpace(\
                (self._adapted_size_factor*X.size), \
                distances=1./X.size)
            rg_domain = ift.makeDomain(ift.RGSpace(X.size))
        else:
            raise NotImplementedError

        self._X, self._Y = \
            ift.makeField(ift.UnstructuredDomain(X.size), X),\
            ift.makeField(ift.UnstructuredDomain(Y.size), Y)
    
        if self.version in ['v1','v2','v3','v4']:
            params = model[self.version]
        else:
            raise NotImplementedError        
        
        self._amp_pdf_x, correlated_fld_pdf_x = \
            get_corr_and_amp(params, 'correlated_field', 'f_X',
                             self._extended_domain, "cause_exp_beta_")
        
        correlated_fld_pdf_x = correlated_fld_pdf_x.exp()
        mask = GeomMaskOperator(correlated_fld_pdf_x.target, rg_domain)
        
        # normalize the exp(f) field in order to have a pdf
        normal = normalize(correlated_fld_pdf_x.target)

        # Mask first then normalize! Now it is a pdf
        self._correlated_fld_pdf_x = \
            normal(mask(correlated_fld_pdf_x))

        cdf = CDF(rg_domain)
        # Need to have cdf in range [0,1]
        rescale = rescalemax(cdf.target)
        cdf = rescale(cdf(self._correlated_fld_pdf_x))

        unis = ift.UniformOperator(rg_domain).ducktape('u_cause')

        # Move to UnstructuredDomain in order to be used
        # for 'op_icdf' below as right 'point_dom'
        GR = ift.GeometryRemover(unis.target)
        unis = GR(unis)

        self._u_cause = unis

        self.op_cdf = ift.FieldAdapter(cdf.target, 'cdf_key').adjoint @ cdf
        self.op_unis = ift.FieldAdapter(unis.target, 'u_key').adjoint @ unis

        op_icdf = CmfLinearInterpolator(
            cdf.target, 'cdf_key',
            unis.target, 'u_key')

        # This ICDF would be the one that should explain the X-data
        # i.e. model the pdf(x)

        self.op_icdf = op_icdf(self.op_cdf + self.op_unis)
        
        alpha = params['noise_scale']['alpha']
        q = params['noise_scale']['q']

        # We take IG prior for sigma^2, hence we need 'one_over' and 'sqrt'
        # domain is just 1D unstructured domain, since sigma is a number

        scalar_domain = ift.DomainTuple.scalar_domain()

        sigma_inv_cause = (ift.InverseGammaOperator(scalar_domain, alpha, q))**(-1)
        sigma_inv_cause = sigma_inv_cause.ducktape("cause_sigma_inv")

        CO = ift.ContractionOperator(self._X.domain, spaces=None)
        self._sigma_inv_cause = CO.adjoint @ sigma_inv_cause

        self._f_op_effect, self._f_ps_effect, self._corr_field_f_effect, \
        self._sigma_inv_effect\
        = \
        self.nonlinresponse_model_setup(params, 'f_Y',
                                      self._extended_domain, X, self._Y, self.infer_noise)

        ln_likelihoods = []
        for f_op, data_field, icov in \
            zip(\
            [self.op_icdf, self._f_op_effect], \
            [self._X, self._Y], \
            [self._sigma_inv_cause, self._sigma_inv_effect]):

            add_data = ift.Adder(-data_field)
            residual = add_data(f_op)

            # The likelihood would be the VariableCovGE since we're also inferring
            # the noise_cov (here just single sigma) which has appropriate metric for
            # this case
            FA_res = ift.FieldAdapter(residual.target, 'residual')
            FA_icov = ift.FieldAdapter(icov.target, 'icov')

            residual_at_icov = FA_res.adjoint @ residual + FA_icov.adjoint @ icov

            ln_likelihoods.append(ift.VariableCovarianceGaussianEnergy(\
                    data_field.domain,
                    'residual',
                    'icov',
                    np.float64) @ residual_at_icov)

        self._Ham = \
            self._initialize_Hamiltonians(\
                [ln_likelihoods[0] + ln_likelihoods[1]])[0]

        # FIXME: Think what is best here, whether to save all necesary variables from parent class
        # here locally, or refer to them with super() command, which would require that the class is
        # initialized here with 'super().__init__(*args)''
    
        self.point_estimates = []
       
    def _k_indx(self, positions):
        k_indx_cause = guess_k_indx(self._sigma_inv_cause, self._amp_pdf_x, positions, self.direction)
        k_indx_effect = guess_k_indx(self._sigma_inv_effect, self._f_ps_effect, positions, self.direction)

        k_indx = max(k_indx_cause, k_indx_effect) 

        if k_indx < int(0.4*self._X.size):
            k_indx = int(0.4*self._X.size)
        
        return k_indx

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            pos = {}
            dom = self._Ham.domain
            for key in dom.keys():
                if not key != 'u_cause':
                    pos[key] = 0.1*ift.from_random(dom[key], 'normal')
                else:
                    pos[key] = ift.from_random(dom[key], 'normal')

            pos = ift.MultiField.from_dict(pos)
            positions.append(pos)

        self._initial_mean = pos
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 2)
        ny = kwargs.pop('ny', 4)
        xsize = kwargs.pop('xsize', 15)
        ysize = kwargs.pop('ysize', 20)

        # Temporarily fix the seed
    
        f_op_cause = []
        u_cause = []
        f_op_effect = []
        
        ps_pdf_cause = []
        f_ps_effect = []
        
        pdf_cause = []
        full_f_effect = []
        
        sigma_inv_cause = []
        sigma_inv_effect = []

        rg = ift.RGSpace(self._sigma_inv_cause.target.shape)
        GR_cause = ift.GeometryRemover(rg)

        rg = ift.RGSpace(self._sigma_inv_effect.target.shape)
        GR_effect = ift.GeometryRemover(rg)

        for pos in positions:
            u_cause.append(self._u_cause.force(pos))
            # Take a look at the deviations of f_op_cause from the 
            # Cause-fld values
            f_op_cause.append(self.op_icdf.force(pos))
            f_op_effect.append(self._f_op_effect.force(pos))
            
            ps_pdf_cause.append(self._amp_pdf_x.force(pos))
            f_ps_effect.append(self._f_ps_effect.force(pos))
            
            pdf_cause.append(self._correlated_fld_pdf_x.force(pos))
            full_f_effect.append(self._corr_field_f_effect.force(pos))

            sigma_inv_cause.append(\
                (GR_cause.adjoint @ (self._sigma_inv_cause**(-1)).sqrt()).force(pos))
            sigma_inv_effect.append(\
                (GR_effect.adjoint @ (self._sigma_inv_effect**(-1)).sqrt()).force(pos))

        # Plot the setup
        plot = myPlot()

        shp = len(f_op_cause)
        markers = list(np.full(shp, None))
        alpha = list(np.full(shp, .8))
        labels = list(np.full(shp, ""))
        scatters = list(np.full(shp, False))

        plot.my_add(f_op_cause, label=labels,
                    xcoord=[x.val for x in u_cause],\
                    marker=markers,  scatter=scatters, alpha=alpha,\
                    title=r'$U-X$ plane samples')
        
        shp = len(f_op_effect)
        markers = list(np.full(shp, None)) + ['o']
        alpha = list(np.full(shp, .8)) + [.3]
        labels = list(np.full(shp, "")) + ["Data"]
        scatters = list(np.full(shp, False)) + [True]

        xcoords = list(np.outer(np.ones(shp), self._X.val)) + [self._X.val]

        plot.my_add(f_op_effect+[self._Y], label=labels,
                    xcoord=xcoords, \
                    scatter=scatters, marker=markers, alpha = alpha,\
                    title=r'$Y-X$ plane samples')

        plot.my_add(pdf_cause,\
            title=r"$\rm{pdf}(x)$")

        plot.my_add(full_f_effect, \
            xmin=-1,xmax=2,ymin=-0.05,ymax=max(self._Y.val)+1,\
            title=r"$\rm{f_Y}$ full")

        plot.my_add(sigma_inv_cause, \
            title=r"$\sigma_{f_X}$")

        plot.my_add(sigma_inv_effect, \
            title=r"$\sigma_{f_Y}$")

        plot.my_add(ps_pdf_cause, title=r'$p(\rm{pdf}(X))$')
        plot.my_add(f_ps_effect, title=r'$p(\rm{f_Y})$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
                       name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):
        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

class Bipartite_model_v3(Causal_Model):
    
    """
    Model with the structure U -> X -> Y 
    """

    def __init__(self, cm):

        super().__init__(cm.direction, [cm.X, cm.Y], cm.config,
                         version=cm.version)

        domain = self.domain
        if self.direction=="X->Y":
            X, Y = cm.X, cm.Y
        elif self.direction=="Y->X":
            Y, X = cm.X, cm.Y
        else:
            raise NotImplementedError

        # By default take the setup of the real model for inference
        model = self.model_dict

        if not isinstance(domain, ift.Domain):
            raise ValueError("self.domain not an ift.Domain")

        # I would need extended domain
        if isinstance(domain, ift.RGSpace):
            # Working with 1D case for the moment
            self._adapted_size_factor = 3
            self._extended_domain = ift.RGSpace(\
                (self._adapted_size_factor*X.size), \
                distances=1./X.size)
            rg_domain = ift.makeDomain(ift.RGSpace(X.size))
        else:
            raise NotImplementedError

        self._X, self._Y = \
            ift.makeField(ift.UnstructuredDomain(X.size), X),\
            ift.makeField(ift.UnstructuredDomain(Y.size), Y)
   
        #####################
        #### CAUSE MODEL ####
        #####################

        if self.version in ['v1','v2','v3','v4']:
            params = model[self.version]
        else:
            raise NotImplementedError        
        
        self._amp_pdf_x, correlated_fld_pdf_x = \
            get_corr_and_amp(params, 'correlated_field', 'f_X',
                             self._extended_domain, "cause_exp_beta_")
        
        correlated_fld_pdf_x = correlated_fld_pdf_x.exp()
        mask = GeomMaskOperator(correlated_fld_pdf_x.target, rg_domain)
        
        # normalize the exp(f) field in order to have a pdf
        normal = normalize(mask.target)

        # Mask first then normalize! Now it is a pdf
        self._correlated_fld_pdf_x = \
            normal(mask(correlated_fld_pdf_x))

        cdf = CDF(rg_domain)
        # Need to have cdf in range [0,1]
        rescale = rescalemax(cdf.target)
        cdf = rescale(cdf(self._correlated_fld_pdf_x))

        unis = ift.UniformOperator(rg_domain).ducktape('u_cause')

        # Move to UnstructuredDomain in order to be used
        # for 'op_icdf' below as right 'point_dom'
        GR = ift.GeometryRemover(unis.target)
        unis = GR(unis)

        self._u_cause = unis

        self.op_cdf = ift.FieldAdapter(cdf.target, 'cdf_key').adjoint @ cdf
        self.op_unis = ift.FieldAdapter(unis.target, 'u_key').adjoint @ unis

        op_icdf = CmfLinearInterpolator(
            cdf.target, 'cdf_key',
            unis.target, 'u_key')

        # This ICDF would be the one that should explain the X-data
        # i.e. model the pdf(x)

        self.op_icdf = op_icdf(self.op_cdf + self.op_unis)
        
        alpha = params['f_X']['noise_scale']['alpha']
        q = params['f_X']['noise_scale']['q']

        # We take IG prior for sigma^2, hence we need 'one_over' and 'sqrt'
        # domain is just 1D unstructured domain, since sigma is a number

        scalar_domain = ift.DomainTuple.scalar_domain()

        sigma_inv_cause = (ift.InverseGammaOperator(scalar_domain, alpha, q))**(-1)
        sigma_inv_cause = sigma_inv_cause.ducktape("cause_sigma_inv")

        CO = ift.ContractionOperator(self._X.domain, spaces=None)
        self._sigma_inv_cause = CO.adjoint @ sigma_inv_cause

        ####################
        ### EFFECT MODEL ###
        ####################
        
        alpha = params['f_Y']['noise_scale']['alpha']
        q = params['f_Y']['noise_scale']['q']

        self._f_ps_effect, self._corr_field_f_effect = \
            get_corr_and_amp(\
                params, 'correlated_field', 'f_Y',
                self._extended_domain, "f_Y_")
        
        # Make the interpolator for learning the y from icdf x and mapping
        self._interpolator = myInterpolator(\
            ift.makeDomain(self._extended_domain), 'f',\
            self.op_icdf.target, 'X',\
            pieces=self._adapted_size_factor,\
            shift=True)
        
        X = ift.FieldAdapter(self.op_icdf.target, 'X').adjoint @ self.op_icdf

        f_op = self._corr_field_f_effect
        f_op = ift.FieldAdapter(f_op.target, 'f').adjoint @ f_op

        self._f_op_effect = self._interpolator(X + f_op)

        scalar_domain = ift.DomainTuple.scalar_domain()

        sigma_inv = (ift.InverseGammaOperator(scalar_domain, alpha, q))**(-1)
        sigma_inv = sigma_inv.ducktape("effect_sigma_inv")

        CO = ift.ContractionOperator(self._Y.domain, spaces=None)
        self._sigma_inv_effect = CO.adjoint @ sigma_inv

        ln_likelihoods = []
        for f_op, data_field, icov in \
            zip(\
            [self.op_icdf, self._f_op_effect], \
            [self._X, self._Y], \
            [self._sigma_inv_cause, self._sigma_inv_effect]):

            add_data = ift.Adder(-data_field)
            residual = add_data(f_op)

            # The likelihood would be the VariableCovGE since we're also inferring
            # the noise_cov (here just single sigma) which has appropriate metric for
            # this case
            FA_res = ift.FieldAdapter(residual.target, 'residual')
            FA_icov = ift.FieldAdapter(icov.target, 'icov')

            residual_at_icov = FA_res.adjoint @ residual + FA_icov.adjoint @ icov

            ln_likelihoods.append(ift.VariableCovarianceGaussianEnergy(\
                    data_field.domain,
                    'residual',
                    'icov',
                    np.float64) @ residual_at_icov)

        # Try with studentT energy for the opicdf likelihood

        # ln_likelihood_cause = ift.StudentTEnergy(self.op_icdf.target, theta=5.) @ self.op_icdf

        # ln_likelihoods.append(ln_likelihood_cause)

        self._Ham = \
            self._initialize_Hamiltonians(\
                [ln_likelihoods[0] + ln_likelihoods[1]])[0]

        # FIXME: Think what is best here, whether to save all necesary variables from parent class
        # here locally, or refer to them with super() command, which would require that the class is
        # initialized here with 'super().__init__(*args)''
    
        self.point_estimates = []
       
    def _k_indx(self, positions):
        k_indx_cause = guess_k_indx(self._sigma_inv_cause, self._amp_pdf_x, positions, self.direction)
        k_indx_effect = guess_k_indx(self._sigma_inv_effect, self._f_ps_effect, positions, self.direction)

        k_indx = max(k_indx_cause, k_indx_effect) 

        if k_indx < int(0.4*self._X.size):
            k_indx = int(0.4*self._X.size)
        
        return k_indx

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            pos = {}
            dom = self._Ham.domain
            for key in dom.keys():
                if not (key in self.op_icdf.domain.keys()) and key != 'u_cause':
                    pos[key] = 0.1*ift.from_random(dom[key], 'normal')
                else:
                    pos[key] = ift.from_random(dom[key], 'normal')

            pos = ift.MultiField.from_dict(pos)
            positions.append(pos)

        self._initial_mean = pos
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 2)
        ny = kwargs.pop('ny', 4)
        xsize = kwargs.pop('xsize', 15)
        ysize = kwargs.pop('ysize', 20)

        # Temporarily fix the seed
    
        f_op_cause = []
        u_cause = []
        f_op_effect = []
        
        ps_pdf_cause = []
        f_ps_effect = []
        
        pdf_cause = []
        full_f_effect = []
        
        sigma_inv_cause = []
        sigma_inv_effect = []

        rg = ift.RGSpace(self._sigma_inv_cause.target.shape)
        GR_cause = ift.GeometryRemover(rg)

        rg = ift.RGSpace(self._sigma_inv_effect.target.shape)
        GR_effect = ift.GeometryRemover(rg)

        for pos in positions:
            u_cause.append(self._u_cause.force(pos))
            # Take a look at the deviations of f_op_cause from the 
            # Cause-fld values
            f_op_cause.append(self.op_icdf.force(pos))
            f_op_effect.append(self._f_op_effect.force(pos))
            
            ps_pdf_cause.append(self._amp_pdf_x.force(pos))
            f_ps_effect.append(self._f_ps_effect.force(pos))
            
            pdf_cause.append(self._correlated_fld_pdf_x.force(pos))
            full_f_effect.append(self._corr_field_f_effect.force(pos))

            sigma_inv_cause.append(\
                (GR_cause.adjoint @ (self._sigma_inv_cause**(-1)).sqrt()).force(pos))
            sigma_inv_effect.append(\
                (GR_effect.adjoint @ (self._sigma_inv_effect**(-1)).sqrt()).force(pos))

        # Plot the setup
        plot = myPlot()

        shp = len(f_op_cause)
        markers = list(np.full(shp, None))
        alpha = list(np.full(shp, .8))
        labels = list(np.full(shp, ""))
        scatters = list(np.full(shp, False))

        plot.my_add(u_cause, label=labels,
                xcoord=[x.val for x in f_op_cause],\
                marker=markers, scatter=scatters, alpha=alpha,\
                title=r'$X-U$ plane samples')
    
        shp = len(f_op_effect)
        markers = list(np.full(shp, None)) + ['o']
        alpha = list(np.full(shp, .8)) + [.3]
        labels = list(np.full(shp, "")) + ["Data"]
        scatters = list(np.full(shp, False)) + [True]

        plot.my_add(f_op_effect+[self._Y], label=labels,
                    xcoord=[x.val for x in f_op_cause] + [self._X.val], \
                    scatter=scatters, marker=markers, alpha = alpha,\
                    title=r'$Y-X$ plane samples')

        plot.my_add(pdf_cause,\
            title=r"$\rm{pdf}(x)$")

        plot.my_add(full_f_effect, \
            xmin=-1,xmax=2.,ymin=-0.05,ymax=max(self._Y.val)+1,\
            title=r"$\rm{f_Y}$ full")

        plot.my_add(sigma_inv_cause, \
            title=r"$\sigma_{f_X}$")

        plot.my_add(sigma_inv_effect, \
            title=r"$\sigma_{f_Y}$")

        plot.my_add(ps_pdf_cause, title=r'$p(\rm{pdf}(X))$')
        plot.my_add(f_ps_effect, title=r'$p(\rm{f_Y})$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
                       name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):
        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)


class Bipartite_model_v4(Causal_Model):
    
    """
    Model with the structure U -> X -> Y 
    """

    def __init__(self, cm):

        super().__init__(cm.direction, [cm.X, cm.Y], cm.config,
                         version=cm.version)

        domain = self.domain
        if self.direction=="X->Y":
            X, Y = cm.X, cm.Y
        elif self.direction=="Y->X":
            Y, X = cm.X, cm.Y
        else:
            raise NotImplementedError

        # By default take the setup of the real model for inference
        model = self.model_dict

        if not isinstance(domain, ift.Domain):
            raise ValueError("self.domain not an ift.Domain")

        # I would need extended domain
        if isinstance(domain, ift.RGSpace):
            # Working with 1D case for the moment
            self._adapted_size_factor = 3
            self._extended_domain = ift.RGSpace(\
                (self._adapted_size_factor*X.size), \
                distances=1./X.size)
            rg_domain = ift.makeDomain(ift.RGSpace(X.size))
        else:
            raise NotImplementedError

        self._X, self._Y = \
            ift.makeField(ift.UnstructuredDomain(X.size), X),\
            ift.makeField(ift.UnstructuredDomain(Y.size), Y)
   
        #####################
        #### CAUSE MODEL ####
        #####################
        
        if self.version in ['v1','v2','v3','v4']:
            params = model[self.version]
        else:
            raise NotImplementedError
        
        ####################
        ### EFFECT MODEL ###
        ####################
        
        self._f_op_effect, self._f_ps_effect, self._corr_field_f_effect, \
        self._sigma_inv_effect\
        = \
        self.nonlinresponse_model_setup(params, 'f',
            self._extended_domain, self._X.val, self._Y, self.infer_noise)

        ln_likelihoods = []
        for f_op, data_field, icov in \
            zip(\
            [self._f_op_effect], \
            [self._Y], \
            [self._sigma_inv_effect]):

            add_data = ift.Adder(-data_field)
            residual = add_data(f_op)

            # The likelihood would be the VariableCovGE since we're also inferring
            # the noise_cov (here just single sigma) which has appropriate metric for
            # this case
            FA_res = ift.FieldAdapter(residual.target, 'residual')
            FA_icov = ift.FieldAdapter(icov.target, 'icov')

            residual_at_icov = FA_res.adjoint @ residual + FA_icov.adjoint @ icov

            ln_likelihoods.append(ift.VariableCovarianceGaussianEnergy(\
                    data_field.domain,
                    'residual',
                    'icov',
                    np.float64) @ residual_at_icov)

        # Try with studentT energy for the opicdf likelihood

        # ln_likelihood_cause = ift.StudentTEnergy(self.op_icdf.target, theta=5.) @ self.op_icdf

        # ln_likelihoods.append(ln_likelihood_cause)

        self._Ham = \
            self._initialize_Hamiltonians(\
                [ln_likelihoods[0]])[0]

        # FIXME: Think what is best here, whether to save all necesary variables from parent class
        # here locally, or refer to them with super() command, which would require that the class is
        # initialized here with 'super().__init__(*args)''
    
        self.point_estimates = []
       
    def _k_indx(self, positions):
        k_indx = guess_k_indx(self._sigma_inv_effect, self._f_ps_effect, positions, self.direction)

        return k_indx

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean                
            pos = 0.1*ift.from_random(self._Ham.domain, 'normal')
            positions.append(pos)

        self._initial_mean = pos
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 2)
        ny = kwargs.pop('ny', 2)
        xsize = kwargs.pop('xsize', 15)
        ysize = kwargs.pop('ysize', 20)

        # Temporarily fix the seed
    
        f_op_effect = []
        
        f_ps_effect = []
        
        full_f_effect = []
        
        sigma_inv_effect = []

        
        rg = ift.RGSpace(self._sigma_inv_effect.target.shape)
        GR_effect = ift.GeometryRemover(rg)

        for pos in positions:
            
            f_op_effect.append(self._f_op_effect.force(pos))
            
            f_ps_effect.append(self._f_ps_effect.force(pos))
            
            full_f_effect.append(self._corr_field_f_effect.force(pos))

            sigma_inv_effect.append(\
                (GR_effect.adjoint @ (self._sigma_inv_effect**(-1)).sqrt()).force(pos))

        # Plot the setup
        plot = myPlot()

        shp = len(f_op_effect)
        markers = list(np.full(shp, None)) + ['o']
        alpha = list(np.full(shp, .8)) + [.3]
        labels = list(np.full(shp, "")) + ["Data"]
        scatters = list(np.full(shp, False)) + [True]

        plot.my_add(f_op_effect+[self._Y], label=labels,
                    xcoord=shp*[self._X.val] + [self._X.val], \
                    scatter=scatters, marker=markers, alpha = alpha,\
                    title=r'$Y-X$ plane samples')

        plot.my_add(full_f_effect, \
            xmin=-1,xmax=2.,ymin=-0.05,ymax=max(self._Y.val)+1,\
            title=r"$\rm{f_Y}$ full")

        plot.my_add(sigma_inv_effect, \
            title=r"$\sigma_{f_Y}$")

        plot.my_add(f_ps_effect, title=r'$p(\rm{f_Y})$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
                       name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):
        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

