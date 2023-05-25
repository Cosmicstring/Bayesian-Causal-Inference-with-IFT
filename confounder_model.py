import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm 

from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


import nifty6 as ift

from operator_utilities import normalize, rescalemax, \
    CmfLinearInterpolator, Confounder_merge, CDF, GeomMaskOperator, \
    myInterpolator

from plotting_utilities import myPlot
from model_utilities import guess_k_indx, get_corr_and_amp, SingleDomain

from causal_model import Causal_Model

class Confounder_model(Causal_Model):

    def __init__(self, cm, \
        merge = 0, 
        factor = 3, verbose=False):

        # I will take this domain to be the
        # domain from which Z-field domain would be
        # created

        super().__init__(cm.direction, [cm.X, cm.Y], cm.config, cm.version)

        self._domain = ift.makeDomain(self.domain)
        
        if not isinstance(self._domain, ift.DomainTuple):
            raise ValueError("self.domain not an ift.Domain")
        
        # FIXME: One has to define the confounder field 'Z'
        # over a domain 3 times bigger than the 'domain', so that
        # 'nonlinear_response' field could be able to ignore the
        # periodic boundary conditions of the RGSpace

        # NOTE: I make a generative model for X <- Z -> Y such that
        # Z ~ InverseCDF(cdf, uniform) for lognormal field, X = f_X(z) + n_X,
        # Y = f_Y(z) + n_Z

        # Make UnstructuredDomains for X,Y
        u_domain = ift.makeDomain(ift.UnstructuredDomain((self.X.size + self.Y.size)))

        data_fld = np.stack((self.X,self.Y)).flatten()
        self._data_fld = ift.makeField(u_domain, data_fld)

        # Add the exponentiated field which is to be normalized
        # and passed to cdf, and plays the role of the unknown
        # pdf of the Z-field

        if self.version in {'v1','v2', 'v4', 'v5'}:
            self._rg_domain = ift.makeDomain(ift.RGSpace(self.X.shape)) # X-Y pairs, same shape
        elif self.version == 'v3':
            self._rg_domain = ift.makeDomain(ift.RGSpace((self.nbins,)))
        else:
            raise NotImplementedError

        # I would need extended domain
        if isinstance(self._domain[0], ift.RGSpace):
            # Working with 1D case for the moment
            adapted_size = factor*self.X.size
            self._adapted_size_factor = factor
            self._extended_domain = ift.makeDomain(ift.RGSpace((adapted_size,),\
                distances=1./self.X.size))
            # FIXME
            # Possibly setting the distances here would mess up something
            #distances=1./X.size))
        else:
            raise NotImplementedError

        self._merge = merge
        self._verbose = verbose
        
    @property
    def rg_domain(self):
        return self._rg_domain
    
    @property
    def extended_domain(self):
        return self._extended_domain

    @property
    def adapted_size_factor(self):
        return self._adapted_size_factor
    

    @property
    def com(self):
        return self._com
    
    @property
    def model(self):
        return self.model_dict

    @property
    def domain(self):
        return self._domain

    @property
    def data_fld(self):
        return self._data_fld
    
    @property
    def merge(self):
        return self._merge

    @property
    def verbose(self):
        return self._verbose

    def _get_Ham(self):

        if self.merge:

            icov_merge = Confounder_merge(\
                self._sigma_inv_X.target, 'sig_X', \
                self._sigma_inv_Y.target, 'sig_Y', \
                self._data_fld.domain)
            
            FA_icov_X = ift.FieldAdapter(self._sigma_inv_X.target, 'sig_X')
            FA_icov_Y = ift.FieldAdapter(self._sigma_inv_Y.target, 'sig_Y')
            input_op = FA_icov_X.adjoint @ self._sigma_inv_X + FA_icov_Y.adjoint @ self._sigma_inv_Y
            
            icov = icov_merge(input_op)

            f_op_merge = Confounder_merge(\
                self._f_X_op.target, 'f_X', \
                self._f_Y_op.target, 'f_Y', \
                self._data_fld.domain)
            
            FA_op_X = ift.FieldAdapter(self._f_X_op.target, 'f_X')
            FA_op_Y = ift.FieldAdapter(self._f_Y_op.target, 'f_Y')
            input_f_op = FA_op_X.adjoint @ self._f_X_op + FA_op_Y.adjoint @ self._f_Y_op

            f_op = f_op_merge(input_f_op)

            # Then I need to construct y-f(x) operator
            # Note that the it is the same as having f(x) - y

            add_data = ift.Adder(-self._data_fld)

            residual = add_data(f_op)

            # The likelihood would be the VariableCovGE since we're also inferring
            # the noise_cov (here just single sigma) which has appropriate metric for
            # this case
            FA_res = ift.FieldAdapter(residual.target, 'residual')
            FA_icov = ift.FieldAdapter(icov.target, 'icov')

            residual_at_icov = FA_res.adjoint @ residual + FA_icov.adjoint @ icov

            ln_likelihood = (ift.VariableCovarianceGaussianEnergy(\
                    self._data_fld.domain,
                    'residual',
                    'icov',
                    np.float64) @ residual_at_icov)

            self._Ham = self._initialize_Hamiltonians([ln_likelihood])[0]
        else:

            ln_likelihood = []
            for data_fld, f_op, icov in zip(\
                [self.fld_X, self.fld_Y], \
                [self._f_X_op, self._f_Y_op], \
                [self._sigma_inv_X, self._sigma_inv_Y]):

                add_data = ift.Adder(-data_fld)

                residual = add_data(f_op)

                # The likelihood would be the VariableCovGE since we're also inferring
                # the noise_cov (here just single sigma) which has appropriate metric for
                # this case
                FA_res = ift.FieldAdapter(residual.target, 'residual')
                FA_icov = ift.FieldAdapter(icov.target, 'icov')

                residual_at_icov = FA_res.adjoint @ residual + FA_icov.adjoint @ icov

                ln_likelihood.append((ift.VariableCovarianceGaussianEnergy(\
                        data_fld.domain,
                        'residual',
                        'icov',
                        np.float64) @ residual_at_icov))

            self._Ham = self._initialize_Hamiltonians([ln_likelihood[0] + ln_likelihood[1]])[0]

        # Setup the keys needed for final plotting
        self.keys = ['f_X', 'f_Y', 'sigma_X', 'sigma_Y']

        # Setup the Hamiltonian for the confounder model
       
        if self.verbose:

            from playground import playground_confounder

            ops = {}
            ops['f_X'], ops['f_Y'] = self._f_X_op, self._f_Y_op
            ops['f_X_ps'], ops['f_Y_ps'] = self._f_X_ps, self._f_Y_ps
            ops['fld_X'], ops['fld_Y'] = self.fld_X, self.fld_Y
            ops['Ham'] = self._Ham
            ops['sigma_inv_X'], ops['sigma_inv_Y'] = self._sigma_inv_X, self._sigma_inv_Y
            ops['corr_fld_X'] = self._corr_field_f_X;
            ops['corr_fld_Y'] = self._corr_field_f_Y;
            ops['cdf'] = cdf
            ops['op_icdf'] = self.op_icdf
            ops['minimizer'] = self.minimizer
            
            if verbose:
                playground_confounder(ops, self._data_fld, self.keys)
            exit()

class Confounder_model_v1(Confounder_model):

    def __init__(self, cm):

        super().__init__(cm)

        rg_domain = self.rg_domain
        extended_domain = self.extended_domain
        model = self.model
        config = self.config
        direction = self.direction

        self._amp_pdf_z, correlated_fld_pdf_z = \
            get_corr_and_amp(model, 'correlated_field', 'Z',
                             extended_domain[0], "Z_exp_beta_")

        # normalize the exp(f) field in order to have a pdf
        mask = GeomMaskOperator(extended_domain, rg_domain)
        normal = normalize(mask.target)
        
        correlated_fld_pdf_z = \
            normal(mask(correlated_fld_pdf_z.exp()))

        self._correlated_fld_pdf_z = correlated_fld_pdf_z

        cdf = CDF(rg_domain)
        # Need to have cdf in range [0,1]
        rescale = rescalemax(cdf.target)
        cdf = rescale(cdf(self._correlated_fld_pdf_z))

        unis = ift.UniformOperator(rg_domain).ducktape('u_xi')

        # Move to UnstructuredDomain in order to be used
        # for 'op_icdf' below as right 'point_dom'
        GR = ift.GeometryRemover(unis.target)
        unis = GR(unis)

        self.op_cdf = ift.FieldAdapter(cdf.target, 'cdf_key').adjoint @ cdf
        self.op_unis = ift.FieldAdapter(unis.target, 'u_key').adjoint @ unis

        op_icdf = CmfLinearInterpolator(
            cdf.target, 'cdf_key',
            unis.target, 'u_key')

        self.op_icdf = op_icdf(self.op_cdf + self.op_unis)

        self._interpolator = myInterpolator(
            extended_domain, 'f', self.op_icdf.target, 'z', \
            pieces = self.adapted_size_factor, \
            shift=True)

        # Initialize likelihoods for X and Y fields

        # FIXME: Of course one would not initialize in this way all the fields which would be necessary
        # for the full causal graph in the future, but for now it is convenient for me to do it this
        # way. If there is a better way, please suggest it, probably one would make a list / dict and
        # iterate through that.

        if self.infer_noise == 0:
                self._f_X_op, self._f_X_ps = \
                self.nonlinresponse_model_setup(model, 'f_X',
                                                extended_domain[0], None, self.fld_X, self.infer_noise,
                                                name='f_X_')
                self._f_Y_op, self._f_Y_ps = \
                self.nonlinresponse_model_setup(model, 'f_Y',
                                                extended_domain[0], None, self.fld_Y, self.infer_noise,
                                                name='f_Y_')
        elif self.infer_noise == 1:
                
                self._f_X_op, self._f_X_ps, self._corr_field_f_X, \
                self._sigma_inv_X \
                = \
                self.nonlinresponse_model_setup(model, 'f_X',
                                                extended_domain[0], None, self.fld_X, self.infer_noise,
                                                name='f_X_')

                self._f_Y_op, self._f_Y_ps, self._corr_field_f_Y, \
                self._sigma_inv_Y \
                = \
                self.nonlinresponse_model_setup(model, 'f_Y',
                                                extended_domain[0], None, self.fld_Y, self.infer_noise,
                                                name='f_Y_')
        elif self.infer_noise == 2:
                self._f_X_op, self._f_X_ps, self._corr_field_f_X, \
                self._sigma_sqr_X \
                = \
                self.nonlinresponse_model_setup(model, 'f_X',
                                                extended_domain[0], None, self.fld_X, self.infer_noise,
                                                name='f_X_')

                self._f_Y_op, self._f_Y_ps, self._corr_field_f_Y, \
                self._sigma_sqr_Y \
                = \
                self.nonlinresponse_model_setup(mode, 'f_Y',
                                                extended_domain[0], None, self.fld_Y, self.infer_noise,
                                                name='f_Y_')

        self._get_Ham()

    def _k_indx(self, positions):
        
        k_indx_X, k_indx_Y = \
        guess_k_indx(self._sigma_inv_X, self._amp_f_x, positions,\
            direction=self.direction, version=self.version), \
        guess_k_indx(self._sigma_inv_Y, self._amp_f_y, positions, \
            direction=self.direction, version=self.version)
        
        return max(k_indx_X, k_indx_Y)

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            mean = {}
            dom = self._Ham.domain
            for key in dom.keys():
                if not (key in self.op_icdf.domain.keys()) and key != 'u':
                    mean[key] = 0.1*ift.from_random(dom[key], 'normal')
                else:
                    mean[key] = ift.from_random(dom[key], 'normal')

            mean = ift.MultiField.from_dict(mean)
            positions.append(mean)

        self._initial_mean = mean
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 3)
        ny = kwargs.pop('ny', 2)
        xsize = kwargs.pop('xsize', 16)
        ysize = kwargs.pop('ysize', 16)

        f_X_list = []
        f_Y_list = []
        f_X_list_unsorted = []
        f_Y_list_unsorted = []
        pdf_Z_list = []
        f_X_ps_list = []
        f_Y_ps_list = []
        z_coord_list = []

        for pos in positions:
        
            # Put the output fields in right order of indices
            # w.r.t. to the z-field

            z = self.op_icdf.force(pos).val
            idx = z.argsort()

            z_coord_list.append(z[idx])

            f_X_op = self._f_X_op.force(pos)
            f_X_list_unsorted.append(f_X_op)
            f_X_op = ift.makeField(f_X_op.domain, f_X_op.val[idx])
            f_X_list.append(f_X_op)

            f_Y_op = self._f_Y_op.force(pos)
            f_Y_list_unsorted.append(f_Y_op)
            f_Y_op = ift.makeField(f_Y_op.domain, f_Y_op.val[idx])
            f_Y_list.append(f_Y_op)

            pdf_Z_list.append(self._correlated_fld_pdf_z.force(pos))
            f_X_ps_list.append(self._f_Y_ps.force(pos))
            f_Y_ps_list.append(self._f_X_ps.force(pos))

        # Plot beta_X setup
        plot = myPlot()

        z_coord_mean = np.mean(np.asarray(z_coord_list),axis=0)

        plot.my_add(\
            f_Y_list +  [self.fld_Y], \
            xcoord= [x.val for x in f_X_list] + [self.fld_X.val], \
            sorted= len(f_Y_list) * [True] + [False], \
            scatter=len(f_Y_list) * [False] + [True], \
            marker= len(f_Y_list) * [None] + ["x"], \
            label= len(f_Y_list) * [""] + ["Data"], \
            title="X - Y plane")

        # FIXME - For some reason the z_coord_list[0] gives
        # the same coordinates as the true ground truth data set
        # in case I do the testing with synthetic data
        plot.my_add(
            f_X_list + [self.fld_X],
            xcoord=z_coord_list + [z_coord_mean], \
            scatter=len(f_X_list) * [False] + [True], \
            marker=len(f_X_list) * [None] + ["x"],
            label= len(f_X_list) * [""] + ["Data"], \
            title="Z - X plane")

        plot.my_add(\
            f_Y_list + [self.fld_Y],
            xcoord= z_coord_list + [z_coord_mean], \
            scatter=len(f_Y_list) * [False] + [True], \
            marker=len(f_Y_list) * [None] + ["x"], \
            label= len(f_Y_list) * [""] + ["Data"], \
            title="Z - Y plane")

        plot.my_add(\
            pdf_Z_list, \
            xcoord=z_coord_list, \
            scatter=len(pdf_Z_list) * [False],\
            marker=len(pdf_Z_list) * [None], \
            label= len(pdf_Z_list) * [""], \
            title=r'$\rm{pdf}_z$')

        plot.my_add(\
            f_X_ps_list, title=r'ps $\rm{f_X}$')
        plot.my_add(\
            f_Y_ps_list, title=r'ps $\rm{f_Y}$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
           name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

class Confounder_model_v2(Confounder_model):

    """
    Trying to model the P(X, Y | Z) = P(X|Z) P(Y|Z) through ICDF
    transform, learning the P(X|Z) and P(Y|Z).

    NOTE: Potential problem could be that this works only for special
    types of mappings, i.e. bijective mappings
    """

    def __init__(self, cm, **kwargs):

        super().__init__(cm)

        rg_domain = self.extended_domain
        model = self.model
        config = self.config
        direction = self.direction

        hat_u = ift.UniformOperator(rg_domain).ducktape('u_xi')

        self._amp_f_x, self._corr_f_x = \
            get_corr_and_amp(\
                model, 'correlated_field', 'f_X', rg_domain[0], 'pdf_f_X_')

        self._amp_f_y, self._corr_f_y = \
            get_corr_and_amp(\
                model, 'correlated_field', 'f_Y', rg_domain[0], 'pdf_f_Y_')
        
        # FIXME: Maybe here for the pdf-fields one needs to take into account that they could
        # don't have to fall down to zero at the edges of the X / Y domains 
        # Maybe another GeomMaskOp here would be useful

        mask = GeomMaskOperator(rg_domain, self.rg_domain)
        # self._corr_f_x = GMO(self._corr_f_x)
        # self._corr_f_y = GMO(self._corr_f_y)

        normal = normalize(mask.target)
        self._pdf_f_x = normal(mask(self._corr_f_x.exp()))

        normal = normalize(mask.target)
        self._pdf_f_y = normal(mask(self._corr_f_y.exp()))

        cdf = CDF(rg_domain)
        rescale = rescalemax(cdf.target)

        cdf_f_x, cdf_f_y = \
            rescale(cdf(self._pdf_f_x)),\
            rescale(cdf(self._pdf_f_y))

        self._cdf_f_x, self._cdf_f_y = cdf_f_x, cdf_f_y

        self.op_cdf_x, self.op_cdf_y = \
            ift.FieldAdapter(cdf_f_x.target, 'cdf_f_x_key').adjoint @ cdf_f_x, \
            ift.FieldAdapter(cdf_f_y.target, 'cdf_f_y_key').adjoint @ cdf_f_y

        # Move to UnstructuredDomain in order to be used
        # for 'op_icdf' below as right 'point_dom'
        GR = ift.GeometryRemover(hat_u.target)
        hat_u = GR(hat_u)

        self.op_unis = ift.FieldAdapter(hat_u.target, 'u_key').adjoint @ hat_u

        op_icdf_f_x, op_icdf_f_y = \
            CmfLinearInterpolator(
                cdf_f_x.target, 'cdf_f_x_key',
                hat_u.target, 'u_key'), \
            CmfLinearInterpolator(
                cdf_f_y.target, 'cdf_f_y_key',
                hat_u.target, 'u_key')

        self._f_X_op, self._f_Y_op = \
            op_icdf_f_x(self.op_cdf_x + self.op_unis),\
            op_icdf_f_y(self.op_cdf_y + self.op_unis)

        if self.version=='v2':
            # Prior for noise -- Assuming same noise_variance for all data points,
            # i.e. learning only one parameter
            alpha = config['real_model'][direction]['noise_scale']['alpha']
            q = config['real_model'][direction]['noise_scale']['q']

            # Make noise-covariances
            scalar_domain = ift.DomainTuple.scalar_domain()

            # FIXME: Note that here I assume the same prior setup (alpha, q values)
            # for both noise variables, but this could be of course adjusted for different
            # priors as well

            # Maybe inverse gamma is not a good prior for noise in this situation
            # since I would like to enforce small noise allowance, because the validity
            # of the model is at question above a certain threshold

            sigma_inv_X, sigma_inv_Y = \
                ((ift.InverseGammaOperator(scalar_domain, alpha, q))**(-1)).ducktape('sigma_X'),\
                ((ift.InverseGammaOperator(scalar_domain, alpha, q))**(-1)).ducktape('sigma_Y')

            # Now to make one single sigma on the whole y_domain

            # NOTE: Same domains for fld_X and fld_Y
            CO = ift.ContractionOperator(self.fld_X.domain, spaces=None)

            self._sigma_inv_X, self._sigma_inv_Y = \
                CO.adjoint @ sigma_inv_X, \
                CO.adjoint @ sigma_inv_Y

            # Now here I make an educated guess for how many eigenvalues I would need to calculate
            # in the BCI_ver4.py : get_evidence(). The number should be roughly equal to the indx of
            # the k-mode where the prior powerspec and noise powerspec intersect

            self._get_Ham()
    
    def _k_indx(self, positions):
        k_indx_X, k_indx_Y = \
            guess_k_indx(self._sigma_inv_X, self._amp_f_x, positions, \
                direction=self.direction, version=self.version), \
            guess_k_indx(self._sigma_inv_Y, self._amp_f_y, positions, \
                direction=self.direction, version=self.version)
            
        return max(k_indx_X, k_indx_Y)

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            mean = 0.1*ift.from_random(self._Ham.domain, 'normal')
            positions.append(mean)

        self._initial_mean = mean
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 3)
        ny = kwargs.pop('ny', 3)
        xsize = kwargs.pop('xsize', 25)
        ysize = kwargs.pop('ysize', 25)

        f_X_list = []
        f_Y_list = []
        pdf_f_X_list = []
        pdf_f_Y_list = []
        full_pdf_f_X_list = []
        full_pdf_f_Y_list = []
        sigma_inv_X_list = []
        sigma_inv_Y_list = []
        f_X_ps_list = []
        f_Y_ps_list = []
        

        ymax_X = 0; ymax_Y = 0
        for pos in positions:
        
            # Put the output fields in right order of indices
            # w.r.t. to the z-field

            f_X_list.append(self._f_X_op.force(pos))
            f_Y_list.append(self._f_Y_op.force(pos))

            pdf_f_X_list.append(self._pdf_f_x.force(pos))
            pdf_f_Y_list.append(self._pdf_f_y.force(pos)) 

            val_X = (self._corr_f_x.exp()).force(pos)
            val_Y = (self._corr_f_y.exp()).force(pos)

            max_X = max(val_X.val)
            max_Y = max(val_Y.val)
            if ymax_X < max_X:
                ymax_X = max_X + 0.1*max_X
            if ymax_Y < max_Y:
                ymax_Y = max_Y + 0.1*max_Y

            full_pdf_f_X_list.append(val_X)
            full_pdf_f_Y_list.append(val_Y)

            sigma_inv_X_list.append((self._sigma_inv_X**(-1)).sqrt().force(pos))
            sigma_inv_Y_list.append((self._sigma_inv_Y**(-1)).sqrt().force(pos))

            f_X_ps_list.append(self._amp_f_x.force(pos))
            f_Y_ps_list.append(self._amp_f_y.force(pos))

        plot = myPlot()
        # Plot beta_X setup
       
        plot.my_add(\
            f_Y_list +  [self.fld_Y], \
            xcoord= [x.val for x in f_X_list] + [self.fld_X.val], \
            #sorted = len(f_Y_list) * [True] + [True], \
            scatter=len(f_Y_list) * [True] + [True], \
            marker= len(f_Y_list) * [None] + ["x"], \
            label= len(f_Y_list) * [""] + ["Data"], \
            title="X - Y plane")

        # FIXME - For some reason the z_coord_list[0] gives
        # the same coordinates as the true ground truth data set
        # in case I do the testing with synthetic data
        plot.my_add(
            pdf_f_X_list,
            title=r"$\rm{pdf(f_X)}$")

        plot.my_add(\
            pdf_f_Y_list,\
            title=r"$\rm{pdf(f_Y)}$")

        plot.my_add(
            full_pdf_f_X_list,
            xmin=-0.5,xmax=1.5,ymin=0.,ymax=ymax_X,\
            title=r"$\rm{pdf(f_X)}$ full")

        plot.my_add(\
            full_pdf_f_Y_list,\
            xmin=-0.5,xmax=1.5,ymin=0.,ymax=ymax_Y,\
            title=r"$\rm{pdf(f_Y)}$ full")

        xcoord = np.linspace(0, 1, sigma_inv_X_list[0].domain.size)
        plot.my_add(
            sigma_inv_X_list,
            xcoord = len(sigma_inv_X_list) * [xcoord],\
            scatter = len(sigma_inv_X_list) * [False],\
            title=r"$\sigma_{\rm{pdf(f_X)}}$")

        plot.my_add(\
            sigma_inv_Y_list,\
            xcoord = len(sigma_inv_Y_list) * [xcoord],\
            scatter = len(sigma_inv_Y_list) * [False],\
            title=r"$\sigma_{\rm{pdf(f_Y)}}$")

        plot.my_add(\
            f_X_ps_list, title=r'ps $\rm{f_X}$')
        plot.my_add(\
            f_Y_ps_list, title=r'ps $\rm{f_Y}$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
           name=filename)

    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

class Confounder_model_v3(Confounder_model_v2):

    """

    Trying to model the P(X,Y | Z) through a poisson likelihood, assuming
    the noise variance in X and Y direction is smaller than the size of
    the bins

    """

    def __init__(self,cm,**kwargs):

        super().__init__(cm)

        input_op = self._f_X_op.ducktape_left('f_X_op') + self._f_Y_op.ducktape_left('f_Y_op')

        MLE = ift.MultiLinearEinsum(input_op.target, 'i,j->ij')
        self._pdf_x_y = MLE(input_op)

        self._data, edges_X, edges_Y = np.histogram2d(self.X, self.Y, bins=[self.nbins, self.nbins])

        centers_X = (edges_X[1:] + edges_X[:-1])*0.5
        centers_Y = (edges_Y[1:] + edges_Y[:-1])*0.5

        data_fld = ift.makeField(self._pdf_x_y.target, self._data.astype(np.int64))

        self._ln_likelihood = ift.PoissonianEnergy(data_fld) @ self._pdf_x_y

        self._k_indx = self._data.size

        self._Ham = self._initialize_Hamiltonians([self._ln_likelihood])[0]

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            mean = 0.1*ift.from_random(self._Ham.domain, 'normal')
            positions.append(mean)

        self._initial_mean = mean
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 4)
        ny = kwargs.pop('ny', 2)
        xsize = kwargs.pop('xsize', 25)
        ysize = kwargs.pop('ysize', 25)

        pdf_X_Z_list = []
        pdf_Y_Z_list = []
        full_pdf_X_Z_list = []
        full_pdf_Y_Z_list = []
        pdf_X_Z_ps_list = []
        pdf_Y_Z_ps_list = []
        
        cdf_f_X_list = []
        cdf_f_Y_list = []

        ymax_X = 0; ymax_Y = 0

        sc_pdf_X_Y_Z = ift.StatCalculator()

        rg = ift.makeDomain(ift.RGSpace(self._pdf_x_y.target.shape))

        GR = ift.GeometryRemover(rg)

        DC = SingleDomain(self._pdf_x_y.target, GR.target)

        rg1 = ift.makeDomain(ift.RGSpace(self._f_X_op.target.shape))
        GR1 = ift.GeometryRemover(rg1)

        for pos in positions:

            val_pdf_x_y_z = (GR.adjoint @ (DC @ self._pdf_x_y)).force(pos)
            sc_pdf_X_Y_Z.add(val_pdf_x_y_z)
            pdf_X_Z_list.append((GR1.adjoint @ self._f_X_op).force(pos))
            pdf_Y_Z_list.append((GR1.adjoint @ self._f_Y_op).force(pos))

            val_X = (self._corr_f_x.exp()).force(pos)
            val_Y = (self._corr_f_y.exp()).force(pos)

            max_X = max(val_X.val)
            max_Y = max(val_Y.val)
            if ymax_X < max_X:
                ymax_X = max_X + 0.1*max_X
            if ymax_Y < max_Y:
                ymax_Y = max_Y + 0.1*max_Y

            full_pdf_X_Z_list.append(val_X)
            full_pdf_Y_Z_list.append(val_Y)

            pdf_X_Z_ps_list.append(self._amp_f_x.force(pos))
            pdf_Y_Z_ps_list.append(self._amp_f_y.force(pos))

            cdf_f_X_list.append(self._cdf_f_x.force(pos))
            cdf_f_Y_list.append(self._cdf_f_y.force(pos))

        plot = myPlot()

        plot.my_add(
            pdf_X_Z_list,
            title=r"$\rm{pdf(f_X)}$")

        plot.my_add(\
            pdf_Y_Z_list,\
            title=r"$\rm{pdf(f_Y)}$")

        plot.my_add(
            full_pdf_X_Z_list,
            xmin=-0.5,xmax=1.5,ymin=0.,ymax=ymax_X,\
            title=r"$\rm{pdf(f_X)}$ full")

        plot.my_add(\
            full_pdf_Y_Z_list,\
            xmin=-0.5,xmax=1.5,ymin=0.,ymax=ymax_Y,\
            title=r"$\rm{pdf(f_Y)}$ full")

        plot.my_add(\
            cdf_f_X_list,\
            title=r"$\rm{cdf(f_Y)}$ full")

        plot.my_add(
            cdf_f_Y_list,
            title=r"$\rm{cdf(f_X)}$ full")

        plot.my_add(\
            pdf_X_Z_ps_list, title=r'ps $\rm{f_X}$')
        plot.my_add(\
            pdf_Y_Z_ps_list, title=r'ps $\rm{f_Y}$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
           name=filename)

        # Plot the joint pdf

        fig = plt.figure(figsize= (15,10))

        gs = gridspec.GridSpec(1, 2)

        ax = fig.add_subplot(gs[0])

        nx, ny = sc_pdf_X_Y_Z.mean.domain[0].shape
        dx, dy = sc_pdf_X_Y_Z.mean.domain[0].distances

        x = np.arange(nx, dtype=np.float64)*dx
        y = np.arange(ny, dtype=np.float64)*dy

        norm = cm.colors.Normalize(\
            vmax=abs(sc_pdf_X_Y_Z.mean.val).max(), \
            vmin=-abs(sc_pdf_X_Y_Z.mean.val).max())
        cmap = cm.RdBu_r

        cntr0=ax.contourf(x,y, sc_pdf_X_Y_Z.mean.val,\
            extent=(0, nx*dx, 0, ny*dy),\
            cmap=cm.get_cmap(cmap,3),
            norm=norm)
        ax.scatter(self.X, self.Y, c='k', alpha=.3, zorder=1)
        ax.set_aspect(1.0/ax.get_data_ratio())

        ax_1 = fig.add_subplot(gs[1], sharey=ax)

        cntr1=ax_1.contourf(x,y, sc_pdf_X_Y_Z.var.val, \
            extent=(0, nx*dx, 0, ny*dy),\
            cmap=cm.get_cmap(cmap,3), \
            norm=norm)
        ax_1.scatter(self.X, self.Y, c='k', alpha=.3, zorder=1)
        ax_1.set_aspect(1.0/ax_1.get_data_ratio())
        
        divider = make_axes_locatable(ax)
        cax_0 = divider.append_axes("right", size="5%", pad=0.05)
        divider = make_axes_locatable(ax_1)
        cax_1 = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(cntr0, cax=cax_0, ax=ax)
        fig.colorbar(cntr1, cax=cax_1, ax=ax_1)

        fig.tight_layout()

        plt.savefig(filename[:-4] + '_pdf_x_y.pdf')
        
        plt.clf()
        plt.cla()
        plt.close()


    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

class Confounder_model_v4(Confounder_model):

    """
    
    Getting to a basis of uniformly distributed Z -> U, and 
    modeling with Gaussian likelihood (assuming Gaussian noise)
    the (X,Y) distribution through mapping:

    X := f_x(U) + N_x , Y := f_z(U) + N_y 

    """

    def __init__(self,cm,**kwargs):

        super().__init__(cm)

        rg_domain = self.rg_domain
        extended_domain = self.extended_domain
        dom_d = self.fld_X.domain # Both fld_X and fld_Y have same domain
        model = self.model
        config = self.config
        direction = self.direction

        hat_u = ift.UniformOperator(rg_domain).ducktape('u_xi')
    
        self.point_estimates = ['u_xi']

        GR = ift.GeometryRemover(hat_u.target)
        hat_u = GR(hat_u)

        self._U = hat_u

        self._amp_f_x, self._corr_f_x = \
            get_corr_and_amp(\
                model, 'correlated_field', 'f_X', extended_domain[0], 'f_X_')

        self._amp_f_y, self._corr_f_y = \
            get_corr_and_amp(\
                model, 'correlated_field', 'f_Y', extended_domain[0], 'f_Y_')

        _interpolator = myInterpolator(\
            extended_domain, 'f', hat_u.target, 'U_z', \
            pieces = self.adapted_size_factor, \
            shift=True)

        _in = self._corr_f_x.ducktape_left('f') + self._U.ducktape_left('U_z')
        self._f_X_op = _interpolator(_in)

        _in = self._corr_f_y.ducktape_left('f') + self._U.ducktape_left('U_z')
        self._f_Y_op = _interpolator(_in)

        sd = ift.DomainTuple.scalar_domain()

        alpha, q = model['noise_scale']['alpha'], model['noise_scale']['q']

        sigma_inv_X, sigma_inv_Y = \
            ((ift.InverseGammaOperator(sd, alpha, q))**(-1)).ducktape('sigma_X'), \
            ((ift.InverseGammaOperator(sd, alpha, q))**(-1)).ducktape('sigma_X')

        CO = ift.ContractionOperator(self.fld_X.domain, spaces=None)
        self._sigma_inv_X = CO.adjoint @ sigma_inv_X
        self._sigma_inv_Y = CO.adjoint @ sigma_inv_Y

        self._get_Ham()

    def _k_indx(self, positions):

        k_indx_X, k_indx_Y = \
            guess_k_indx(self._sigma_inv_X, self._amp_f_x, positions, \
                direction=self.direction, version=self.version), \
            guess_k_indx(self._sigma_inv_Y, self._amp_f_y, positions, \
                direction=self.direction, version=self.version)

        self._k_indx = max(k_indx_X, k_indx_Y)

#        if self._k_indx < self.fld_X.size:
#            self._k_indx = self.fld_X.size
 
        if self._k_indx > self._Ham.domain.size:
            self._k_indx = self._Ham.domain.size-1
            raise Warning(\
                "k_indx larger than the Hamiltonian domain!"
                "Set the value to Ham.domain.size - 1")
        
        return max(k_indx_X, k_indx_Y)

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            mean = ift.from_random(self._Ham.domain, 'normal')
            positions.append(mean)

        self._initial_mean = mean
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 3)
        ny = kwargs.pop('ny', 3)
        xsize = kwargs.pop('xsize', 16)
        ysize = kwargs.pop('ysize', 16)

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
     
        for pos in positions:
        
            # Put the output fields in right order of indices
            # w.r.t. to the z-field

            u = self._U.force(pos).val
            idx = u.argsort()

            U_list.append(u[idx])

            f_X_op = (self._f_X_op).force(pos)
            f_X_list_unsorted.append(f_X_op)
            f_X_list.append(\
                ift.makeField(f_X_op.domain,f_X_op.val[idx]))

            f_Y_op = (self._f_Y_op).force(pos)
            f_Y_list_unsorted.append(f_Y_op)
            f_Y_list.append(\
                ift.makeField(f_Y_op.domain, f_Y_op.val[idx]))

            f_X_ps_list.append(self._amp_f_x.force(pos))
            f_Y_ps_list.append(self._amp_f_y.force(pos))

            full_X.append(self._corr_f_x.force(pos))
            full_Y.append(self._corr_f_y.force(pos))

            sigma_inv_X_list.append((self._sigma_inv_X**(-1)).sqrt().force(pos))
            sigma_inv_Y_list.append((self._sigma_inv_Y**(-1)).sqrt().force(pos))

        # Plot beta_X setup
        plot = myPlot()

        plot.my_add(\
            f_Y_list + [self.fld_Y], \
            xcoord= [x.val for x in f_X_list] + [self.fld_X.val], \
            sorted= len(f_Y_list) * [True] + [False], \
            scatter=len(f_Y_list) * [False] + [True], \
            marker= len(f_Y_list) * [None] + ["x"], \
            label= len(f_Y_list) * [""] + ["Data"], \
            xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05,\
            title="X - Y plane")
        
        plot.my_add(\
            f_X_list, \
            xcoord = U_list, \
            xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_x}$ plane")

        plot.my_add(\
            f_Y_list, \
            xcoord = U_list,\
            xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_y}$ plane")

        plot.my_add(\
            full_X, \
            xmin=-1.05, xmax=2.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_x}$ plane")

        plot.my_add(\
            full_Y, \
            xmin=-1.05, xmax=2.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_y}$ plane")

        plot.my_add(\
            f_X_ps_list, title=r'ps $\rm{f_X}$')
        plot.my_add(\
            f_Y_ps_list, title=r'ps $\rm{f_Y}$')

        xcoord = np.linspace(0, 1, sigma_inv_X_list[0].domain.size)
        plot.my_add(
            sigma_inv_X_list,
            xcoord = len(sigma_inv_X_list) * [xcoord],\
            scatter = len(sigma_inv_X_list) * [False],\
            title=r"$\sigma_{\rm{(f_X)}}$")

        plot.my_add(\
            sigma_inv_Y_list,\
            xcoord = len(sigma_inv_Y_list) * [xcoord],\
            scatter = len(sigma_inv_Y_list) * [False],\
            title=r"$\sigma_{\rm{(f_Y)}}$")

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
           name=filename)


    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)

class Confounder_model_v5(Confounder_model_v4):

    """
    
    Getting to a basis of uniformly distributed Z -> U, and 
    modeling with Gaussian likelihood (assuming Gaussian noise)
    the (X,Y) distribution through mapping:

    X := f_x(U) + N_x , Y := f_z(U) + N_y 

    """

    def __init__(self,cm,**kwargs):

        super().__init__(cm)

        self._sigma_inv_X, self._sigma_inv_Y = \
            ((ift.ScalingOperator(self.fld_X.domain, 1e-3)).inverse), \
            ((ift.ScalingOperator(self.fld_Y.domain, 1e-3)).inverse)

        # Generate the Hamiltonian, now with fixed noise
        # at a very low level
        ln_likelihood = []
        for data_fld, f_op, icov in zip(\
            [self.fld_X, self.fld_Y], \
            [self._f_X_op, self._f_Y_op], \
            [self._sigma_inv_X, self._sigma_inv_Y]):

            ln_likelihood.append((ift.GaussianEnergy(\
                    mean=data_fld,
                    inverse_covariance=icov) @ f_op))

        self._Ham = self._initialize_Hamiltonians([ln_likelihood[0] + ln_likelihood[1]])[0]

    def _k_indx(self, positions):
        
        k_indx_X, k_indx_Y = \
            guess_k_indx(self._sigma_inv_X, self._amp_f_x, positions, \
                direction=self.direction, version=self.version), \
            guess_k_indx(self._sigma_inv_Y, self._amp_f_y, positions, \
                direction=self.direction, version=self.version)

        self._k_indx = max(k_indx_X, k_indx_Y)

#        if self._k_indx < self.fld_X.size:
#            self._k_indx = self.fld_X.size

        if self._k_indx > self._Ham.domain.size:
            self._k_indx = self._Ham.domain.size-1
            raise Warning(\
                "k_indx larger than the Hamiltonian domain!"
                "Set the value to Ham.domain.size - 1")
        
        return max(k_indx_X, k_indx_Y)

    def plot_initial_setup(self, filename, **kwargs):

        positions = []
        for i in range(10):
            # Initialize the mean
            mean = ift.from_random(self._Ham.domain, 'normal')
            positions.append(mean)

        self._initial_mean = mean
        self._plot_setup(filename.format("prior_samples"), positions, **kwargs)

    def _plot_setup(self, filename, positions, **kwargs):

        nx = kwargs.pop('nx', 3)
        ny = kwargs.pop('ny', 3)
        xsize = kwargs.pop('xsize', 16)
        ysize = kwargs.pop('ysize', 16)

        f_X_list = []
        f_X_list_unsorted = []
        f_Y_list = []
        f_Y_list_unsorted = []
        full_X = []
        full_Y = []
        U_list = []
        f_X_ps_list = []
        f_Y_ps_list = []
        
        for pos in positions:
        
            # Put the output fields in right order of indices
            # w.r.t. to the z-field

            u = self._U.force(pos).val
            idx = u.argsort()

            U_list.append(u[idx])

            f_X_op = self._f_X_op.force(pos)
            f_X_list_unsorted.append(f_X_op)
            f_X_list.append(\
                ift.makeField(f_X_op.domain,f_X_op.val[idx]))

            f_Y_op = self._f_Y_op.force(pos)
            f_Y_list_unsorted.append(f_Y_op)
            f_Y_list.append(\
                ift.makeField(f_Y_op.domain, f_Y_op.val[idx]))

            full_X.append(self._corr_f_x.force(pos))
            full_Y.append(self._corr_f_y.force(pos))

            f_X_ps_list.append(self._amp_f_x.force(pos))
            f_Y_ps_list.append(self._amp_f_y.force(pos))

        # Plot beta_X setup
        plot = myPlot()

        plot.my_add(\
            f_Y_list + [self.fld_Y], \
            xcoord= [x.val for x in f_X_list] + [self.fld_X.val], \
            sorted= len(f_Y_list) * [True] + [False], \
            scatter=len(f_Y_list) * [False] + [True], \
            marker= len(f_Y_list) * [None] + ["x"], \
            label= len(f_Y_list) * [""] + ["Data"], \
            xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05,\
            title="X - Y plane")
        
        plot.my_add(\
            f_X_list, \
            xcoord = U_list, \
            xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_x}$ plane")

        plot.my_add(\
            f_Y_list, \
            xcoord = U_list,\
            xmin=-0.05, xmax=1.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_y}$ plane")

        plot.my_add(\
            full_X, \
            xmin=-1.05, xmax=2.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_x}$ plane")

        plot.my_add(\
            full_Y, \
            xcoord = U_list,\
            xmin=-1.05, xmax=2.05, ymin=-0.05, ymax=1.05,\
            title=r"$U - \rm{f_y}$ plane")

        plot.my_add(\
            f_X_ps_list, title=r'ps $\rm{f_X}$')
        plot.my_add(\
            f_Y_ps_list, title=r'ps $\rm{f_Y}$')

        plot.my_output(ny=ny, nx=nx, xsize=xsize, ysize=ysize,
           name=filename)


    def optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        return self._optimize_and_get_evidence(N_samples, N_steps, **kwargs)
