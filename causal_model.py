import numpy as np

import nifty6 as ift

from model_utilities import guess_k_indx, get_corr_and_amp,\
    save_KL_position, save_random_state, save_KL_sample, Bin

from operator_utilities import GeomMaskOperator
from minimization import stages

class Causal_Model(object):

    def __init__(self, direction, data=None, config=None, version = 'v1'):
        """
        direction : Inference direction, possible are "X->Y", "Y->X", "X<-Z->Y"
        data : List of np.arrays, which are containing information about the X,Y data
                   must be len(data) == 2
        config : config.json file which contains the params for the model setup
        (TO BE ADDED)
        """

        if len(data) == 2:
            self._X, self._Y = data[0], data[1]
            if self._X.shape != self._Y.shape:
                raise ValueError("X-Y pairs are not of the same shape")
        else:
            raise TypeError("Data not in right format")

        self._fld_X = ift.makeField(ift.makeDomain(ift.UnstructuredDomain(self._X.size)), self._X)
        self._fld_Y = ift.makeField(ift.makeDomain(ift.UnstructuredDomain(self._Y.size)), self._Y)

        if (direction == "X->Y") or (direction == "Y->X") \
                or (direction == "X<-Z->Y") or (direction == "X || Y"):
            self._direction = direction
        else:
            raise ValueError("Not Implemented")

        self._version = version

        if config == None:
            self._nbins = 256
            self._shape = (self._nbins,)
            # Set up the domains for the lognormal field and for the corresponding
            # nonlinear response field

            self._domain = ift.RGSpace(self.shape)
            self._config = None
        else:
            # Here one would read in the details of the nbins from the config.json
            # file, I just need to think still where to put the nbins information
            self._config = config

            if self._direction == "X->Y" or self._direction == "Y->X":
                self._model_dict = config["real_model"]["bivariate"]
                self._seed = self._config['real_model']["bivariate"]['seed']
            else:
                self._model_dict = config["real_model"][self._direction]
                self._seed = self._config['real_model'][self._direction]['seed']

            # By default assume the data is not mock_generated, therefore
            # take the params for real data inference
            self._nbins = config['real_model']['Nbins']

            # We are currently working with 1D
            self._domain = ift.RGSpace(self._nbins)

            # Set the seed
            ift.random.push_sseq_from_seed(self._seed)

    @property
    def domain(self):
        return self._domain

    @property
    def model_dict(self):
        return self._model_dict

    @property
    def infer_noise(self):
        return self.model_dict["infer_noise"]

    @property
    def config(self):
        return self._config

    @property
    def seed(self):
        return self._seed

    @property
    def direction(self):
        return self._direction

    @property
    def version(self):
        return self._version

    @property
    def nbins(self):
        return self._nbins

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def fld_X(self):
        return self._fld_X

    @property
    def fld_Y(self):
        return self._fld_Y

    @property
    def minimizer(self):
        return self._minimizer

    @property
    def fail_dictionary(self):
        fail_dict = \
                {
                    "mean" : -np.inf,
                    "upper" : -np.inf,
                    "lower" : -np.inf,
                    "H_lh" : -np.inf,
                    "var_H_lh" : -np.inf,
                    "xi2" : -np.inf,
                    "Tr_reduce_Lambda" : -np.inf,
                    "err_TrL" : -np.inf,
                    "Tr_ln_Lambda" : -np.inf,
                    "err_TrlnL" : -np.inf
                }
        return fail_dict


    def _initialize_numerics(self):

        if self.config == None:
            # Minimization parameters
            self.ic_sampling = ift.AbsDeltaEnergyController(
                deltaE=0.1, iteration_limit=250)
            self.ic_newton = ift.AbsDeltaEnergyController(
                deltaE=1e-5, iteration_limit=5, \
                name='newton', convergence_level=3)
            self._minimizer = ift.NewtonCG(self.ic_newton)
        else:
            config = self.config
            # Params will be taken from the real model
            numerics = config['real_model']['numerics']
            self.ic_sampling = ift.AbsDeltaEnergyController(
                deltaE=numerics['ic_sampling']['deltaE'],
                iteration_limit=numerics['ic_sampling']['iteration_limit'],
                convergence_level=3)
            self.ic_newton = ift.AbsDeltaEnergyController(
                name="Newton",
                deltaE=numerics['ic_newton']['deltaE'],
                iteration_limit=numerics['ic_newton']['iteration_limit'],
                convergence_level=3)
            self._minimizer = ift.NewtonCG(self.ic_newton)

    def _initialize_Hamiltonians(self, ln_likelihoods):

        Hamiltonians = []

        if not isinstance(ln_likelihoods, list):
            ln_likelihoods = [ln_likelihoods]

        self._initialize_numerics()

        for ln_likelihood in ln_likelihoods:
            H = ift.StandardHamiltonian(
                ln_likelihood, ic_samp=self.ic_sampling)
            Hamiltonians.append(H)

        return tuple(Hamiltonians)

    def _setup_cause_effect_flds(self, X, Y):

        domain = self.domain
        counts_domain = ift.UnstructuredDomain(domain.shape)
        Y_domain = ift.UnstructuredDomain(Y.shape)

        if not isinstance(domain, ift.Domain):
            raise ValueError("self.domain not an ift.Domain")

        X_binned_data = Bin(domain, X)

        # Make a field of the binned data
        if isinstance(X_binned_data, np.ndarray):
            Counts_fld = ift.Field.from_raw(counts_domain, X_binned_data)
        else:
            raise ValueError("Not Implemented")

        if not isinstance(Y, ift.Field):
            # Assuming the data is an UnstructuredDomain
            Effect_fld = ift.Field.from_raw(Y_domain, Y)
        else:
            Effect_fld = Y

        return Counts_fld, Effect_fld

    def lognormal_model_setup(self,
                              model, ps_key,
                              extended_domain,
                              cause_fld, name='beta_'):

        # Setting up the hyperparams for the model
        if self.config == None:
            offset_amplitude_mean = 1e-2
            offset_amplitude_stddev = 1e-5
            fluctuations_mean = 0.1
            fluctuations_stddev = 1e-2
            flexibility_mean = 0.1
            flexibility_stddev = 0.05
            asperity_mean = 0.01
            asperity_stddev = 0.05
            loglogavgslope_mean = -4.0
            loglogavgslope_stddev = 0.5
        else:

            ps_flag = model[ps_key]['ps_flag']
            amp_beta, correlated_field_beta = \
                get_corr_and_amp(model, ps_flag, ps_key, extended_domain, name)

        _lamb = ift.exp(correlated_field_beta)
        mask = GeomMaskOperator(_lamb.target, cause_fld.domain)

        lamb = mask(_lamb)

        # Response as ift.GeometryRemover in order to
        # map to UnstructuredDomain
        R = ift.GeometryRemover(lamb.target)
        R_lamb = R(lamb)

        # Actually the cause_domain.domain should always be UnstructuredDomain
        # therefore I need to switch to UnstructuredDomain field
        if not isinstance(cause_fld.domain[0], ift.UnstructuredDomain):
            import warnings

            warnings.warn("Domain of cause is not ift.UnstructuredDomain")
            warnings.warn("Switching to Unstrcutured")
            domain_Unstructured = ift.UnstructuredDomain(cause_fld.shape)
            cause_fld = ift.Field.from_raw(domain_Unstructured, cause_fld.val)
        else:
            pass

        ln_likelihood_beta = ift.PoissonianEnergy(cause_fld)(R_lamb)

        return ln_likelihood_beta, R_lamb, correlated_field_beta, amp_beta

    def nonlinresponse_model_setup(self, \
                                   model, ps_key, \
                                   extended_domain, \
                                   X, effect_fld, infer_noise, name='f_'):
        """
        Note that if isinstance(X, np.ndarray) then one does ift.LinearInterpolator, but
        if X==None, that one assumes an interpolator of type :class: ift.Operator is given
        which would interpolate non-lin response op at given locations-field
        """

        # By default take the setup of the real model for inference
        ps_flag = model[ps_key]['ps_flag']

        amp_f, correlated_field_f = \
            get_corr_and_amp(model, ps_flag, ps_key, extended_domain, name)

        f_op = correlated_field_f

        if isinstance(X, np.ndarray):

            # For the f-field now mask the regions which are outside the
            # data space

            # FIXME: Possibly of better use is to take the
            # myInterpolator here and just interpolate at
            # points where X is defined. This would automatically
            # give me the masking!

            mask = GeomMaskOperator(f_op.target, effect_fld.domain)

            f_op = mask(f_op)

            # Make the interpolator's domain with distances from original extended_domain
            # which is padded
            interpolator_domain = ift.RGSpace(X.shape,
                                              distances=1./(f_op.target.size-1))

            interpolator = ift.LinearInterpolator(
                interpolator_domain, X.reshape(1, -1))

            # For switching between domains
            GR = ift.GeometryRemover(interpolator_domain)

            f_op = interpolator(GR.adjoint(f_op))

        elif X == None:

            # In this case the masking is done for me through
            # the use of interpolator since I am just interpolating
            # the f-field at these spots!

            # Now X would be defined from icdf model
            icdf = self.op_icdf
            interp = self._interpolator

            X = ift.FieldAdapter(icdf.target, 'z').adjoint @ icdf

            f_op = ift.FieldAdapter(f_op.target, 'f').adjoint @ f_op

            f_op = interp(X + f_op)

        else:
            raise NotImplementedError

        # By default take the setup of the real model for inference
        if infer_noise == 0:

            noise_scale = 0.1
            N = ift.ScalingOperator(f_op.target, noise_scale)

            return f_op, amp_f

        elif infer_noise == 1:

            # Prior for noise -- Assuming same noise_variance for all data points,
            # i.e. learning only one parameter
            alpha = model['noise_scale']['alpha']
            q = model['noise_scale']['q']

            # We take IG prior for sigma^2, hence we need 'one_over' and 'sqrt'
            # domain is just 1D unstructured domain, since sigma is a number

            scalar_domain = ift.DomainTuple.scalar_domain()

            sigma_inv = (ift.InverseGammaOperator(scalar_domain, alpha, q))**(-1)
            sigma_inv = sigma_inv.ducktape(name + "sigma_inv")

            CO = ift.ContractionOperator(effect_fld.domain, spaces=None)
            sigma_inv = CO.adjoint @ sigma_inv

            # Now here I make an educated guess for how many eigenvalues I would need to calculate
            # in the BCI_ver4.py : get_evidence(). The number should be roughly equal to the indx of
            # the k-mode where the prior powerspec and noise powerspec intersect

            return f_op, amp_f, correlated_field_f, sigma_inv

        elif infer_noise == 2:

            # TODO: Write up on overleaf all the relevant points

            alpha = model['noise_scale']['alpha']
            q = model['noise_scale']['q']

            # Here I use the MAP solution of the Hamiltonian for the noise_std^2
            # then the H(y,f(x)) has the form of a studentT distribution. Look at
            # my notes for the full expression and derivation at page 27 for BCI

            # After using MAP for sigma_inv theta for studentT becomes this
            # look at bottom of page 27 of BCI

            theta = 2*alpha + effect_fld.size - 1.

            # Since we integrated out the noise we use the studentT with theta
            # given above and f given with:

            add_data = ift.Adder(-effect_fld)

            f = add_data(f_op)

            # Dont forget the prefactor coming from the variable change
            # look at bottom of page 27 of BCI

            print("CHECKING STUDENT_T")

            # From wiki for scale-inv chi^2 prior
            s_sqr = np.sum((effect_fld.val - np.mean(effect_fld.val))
                           ** 2 / (effect_fld.size - 1))
            prefac = np.sqrt(effect_fld.size/s_sqr)
            print(prefac)

            prefac = ift.ScalingOperator(f.target, 10.)

            f = prefac(f)

            ln_likelihood_f = ift.StudentTEnergy(f.target, 1.5)(f)

            # prefac for sigma_sqr

            factor = alpha + effect_fld.size / 2.

            # Add q

            add_q = ift.Adder(ift.Field.full(f.target, q))
            sigma_sqr = add_q(.5*f**2)
            sigma_sqr = factor * sigma_sqr

            return ln_likelihood_f, f_op, amp_f, sigma_sqr

        else:
            raise ValueError("Not Implemented")

    def _get_evidence(self,\
                      KL, \
                      N_resample = 1000, \
                      n_eigs=20, fudge_factor=30, \
                      max_iter = 3,\
                      eps = 1e-4):

        if not isinstance(KL, ift.MetricGaussianKL):
            raise ValueError("KL not ift.MetricGaussianKL")

        # KL.metric is Theta^{-1}!
        metric = KL.metric
        ln_likelihood = KL._hamiltonian._lh

        xi_bar = KL.position

        """
         NOTE: The number 10 was chosen because it was good enough for my problems,
         but you should try to see how many at least one should calculate for your
         problem at hand. One should select 'n_eigs' after looking at the prior correlation
         structure and noise covariance. Think about the Wiener filter case:

        D^{-1} = R^{\dagger} N^{-1} R + S^{-1}

         Hence, one has in the Fourier space (assume R=1)

        D^{-1} ~ P_N ^ {-1} + P_S ^ {-1}

         with P_N and P_S being the power spectrum of noise and prior. Therefore,
         one can get a good guess for the 'n_eigs' by looking at these power spectra.
         Since in the case of MGVI one takes standardized prior, P_S^{-1} == 1, so one
         has for D:

        D^{-1} ~ F^{-1} + 1

         with F being the Fisher metric of the problem. Hence, one needs to look at the
         Fisher metric structure to figure this out.

        """
        N_eigs = n_eigs + fudge_factor
        err_eigs = np.inf
        count = 0
        limit = KL.metric.domain.size - 1
        while (err_eigs > eps) and (count<max_iter) and (2*N_eigs<limit):
            print("Number of eigenvalues to compute")
            print(N_eigs)
            print("\n")

            eigs = ift.operator_spectrum(metric, k=N_eigs, hermitian=True)
            err_eigs = abs(1.0 - eigs[-1])
            # Double the number of eigenvalues
            N_eigs = 2*N_eigs
            count +=1
        print("Check convergence of eigenvalues")
        print(eigs)
        print("\n")

        # Asses the potential error for the eigenvalues:

        max_eigs = metric.domain.size

        min_eig = min(eigs)

        # Calculate the \Tr \ln term and \Tr term
        Tr_reduced_Lambda_Theta = 0
        Tr_ln_Lambda_Theta = 0

        for eig in eigs:
            # The eigenvalues are of Theta^{-1} and
            # we need the 1/eig for the ELBO calculation
            lambda_theta = 1./eig
            if abs(lambda_theta-1.) > eps:
                Tr_reduced_Lambda_Theta += lambda_theta - 1.
            if abs(np.log(lambda_theta)) > eps:
                Tr_ln_Lambda_Theta += np.log(lambda_theta)

        # Propagate the eigs_err for the Tr_Lambda and Tr_ln_Lambda

        delta_n = (max_eigs - N_eigs)
        err_Tr_reduced_Lambda_Theta = delta_n * (min_eig - 1.)
        err_Tr_ln_Lambda_Theta = delta_n * np.log(min_eig)

        # Now calculate the contribution from the prior

        _xi_sqrd = xi_bar.vdot(xi_bar)
        xi_sqrd = _xi_sqrd.val
        prior_evidence = 0.5*(xi_sqrd + Tr_reduced_Lambda_Theta)

        # Get the sampled likelihood

        sampled_sum_ln_likelihood = ift.Field.scalar(0)

        """
         NOTE: the 'sampled_sum_ln_prior' is not really used in
         the calculations, but I was just curious to see whether
         sampling of the prior likelihood would converge to the
         analytically calculated value
        """

        # Now, resample the KL in order to have better view of
        # the surroundings

        _KL_resampled = ift.MetricGaussianKL(
                    KL.position, KL._hamiltonian, N_resample, mirror_samples=True)

        # sampled_sum_ln_prior = ift.Field.scalar(0)

        sc_lnl = ift.StatCalculator()

        for sample in _KL_resampled.samples:
            xi = sample + xi_bar
            sc_lnl.add(ln_likelihood(xi))

            # sampled_sum_ln_prior = sampled_sum_ln_prior + ln_prior(xi)

        mean_ln_likelihood = \
                sc_lnl.mean.val

        if self.direction == "X<-Z->Y" or \
                ((self.direction=="X->Y" or self.direction=="Y->X") and self.version=="v4"):
            # For the confounder model there is one extra 'N/2 \ln 2\Pi' term w.r.t.
            # to the the X->Y and Y->X models, which needs to be substracted
            # here. The prefactor 'N' is number of data d.o.f.

            # NOTE: Here the '+' sign stands, but nonetheless this penalizes
            # the evidence below since the likelihood term going in is
            # '-mean_ln_likelihood'
            mean_ln_likelihood = mean_ln_likelihood + 0.5*self.X.size*np.log(2*np.pi)

        var_ln_likelihood = \
                np.sqrt(sc_lnl.var.val)

        # avrgd_ln_prior = sampled_sum_ln_prior.val / len(KL_samples)

        """
         For the details about this formula take a look at section 4.2 in my overleaf notes:
         https://www.overleaf.com/read/vnpxnhbsbtbm
        """

        evid_mean = (- mean_ln_likelihood - prior_evidence + 0.5*Tr_ln_Lambda_Theta)
        evid_var_upper = abs(var_ln_likelihood  - 0.5*err_Tr_reduced_Lambda_Theta + 0.5*err_Tr_ln_Lambda_Theta)
        evid_var_lower = abs(-var_ln_likelihood - 0.5*err_Tr_reduced_Lambda_Theta + 0.5*err_Tr_ln_Lambda_Theta)

        outp = \
        {
            "mean" : evid_mean,
            "upper" : evid_mean + evid_var_upper,
            "lower" : evid_mean - evid_var_lower,
            "H_lh" : mean_ln_likelihood,
            "var_H_lh" : var_ln_likelihood,
            "xi2" : xi_sqrd,
            "Tr_reduce_Lambda" : Tr_reduced_Lambda_Theta,
            "err_TrL" : err_Tr_reduced_Lambda_Theta,
            "Tr_ln_Lambda" : Tr_ln_Lambda_Theta,
            "err_TrlnL" : err_Tr_ln_Lambda_Theta
        }

        print("Calculated terms")

        print("H_lh: {:.5e} +- {:.5e}\n".format(mean_ln_likelihood, var_ln_likelihood))
        print("xi^2 term: {:.5e} \n".format(xi_sqrd))
        print("Tr \Lambda: {:.5e} (+ {:.5e})\n".format(Tr_reduced_Lambda_Theta, err_Tr_reduced_Lambda_Theta))
        print("Tr \ln \Lambda: {:.5e} (+ {:.5e})\n".format(Tr_ln_Lambda_Theta, err_Tr_ln_Lambda_Theta))
        print("\n")

        return outp


    def _optimize_and_get_evidence(self, N_samples, N_steps, **kwargs):

        minimizer = self.minimizer

        # Set the stages
        stage_1, stage_2, stage_3 = stages(N_steps)

        track_optimization = kwargs.pop("track_optimization", False)
        plot_final = kwargs.pop("plot_final", False)
        pe_keys = kwargs.pop("point_estimates", [])

        if plot_final:
            current_output_path = kwargs.pop('current_output_path', '')
            filename = kwargs.pop('filename', '')

            if current_output_path == '':
                raise ValueError("Need to provide output path")
            if filename == '':
                raise ValueError("Need to provide filename for output")

        # In the case of the 'X->Y' and 'Y->X' models the inference
        # decouples, therefore we need to split the evidence calculation too

        evidence = []
        Ham = self._Ham
        mean = self._initial_mean

        seed = self.seed

        # Random state file
        f_rstate_ID = current_output_path.format(\
                        "{}_rstate_version_{}_seed_{}.txt".format(\
                        self.direction, self.version, seed))

        # Position file
        KL_position_f_ID = current_output_path.format(\
                        "{}_KL_position_version_{}_{}".format(\
                        self.direction, self.version, seed))

        # Sample file
        KL_sample_f_ID = current_output_path.format(\
                        "samples/{}_KL_sample_version_{}_{}_{}".format(\
                        self.direction, self.version, seed, '{:d}'))

        step = np.floor(0.1 * N_steps) + 1

        # One would need tmp sample number for the
        # adaptive sampling

        local_N_samples = N_samples

        # Minimize KL during N_steps total steps
        for i in range(N_steps):

            if i < stage_1:
                local_N_samples = N_samples
            elif i< stage_2:
                local_N_samples = 2*N_samples
                # Stop the MAP estimate here and
                # resample everything
                pe_keys = []
            elif i < stage_3:
                local_N_samples = 4*N_samples

                ic_newton = ift.AbsDeltaEnergyController(deltaE=1e-5, iteration_limit=10, name='newton', convergence_level=3)
                ic_newton.enable_logging()
                minimizer = ift.NewtonCG(ic_newton, enable_logging=True)
            else:
                ic_newton = ift.AbsDeltaEnergyController(deltaE=1e-5, iteration_limit=20, name='newton', convergence_level=3)
                ic_newton.enable_logging()
                minimizer = ift.NewtonCG(ic_newton, enable_logging=True)

            # Draw new samples and minimize KL
            KL = ift.MetricGaussianKL(
                mean, Ham, local_N_samples, point_estimates=pe_keys, nanisinf=True, mirror_samples=True)

            KL, convergence = minimizer(KL)

            mean = KL.position

            # Save the random state for higher quality plots later on
            save_random_state(f_rstate_ID)

            # Save the Position
            save_KL_position(mean.val, KL_position_f_ID)

            # Save the samples
            sample_ID = 0
            for s in KL.samples:
                save_KL_sample(s.val, KL_sample_f_ID.format(sample_ID))
                sample_ID += 1

            if track_optimization:
                # Plot current reconstruction
                if i % step == 0:

                    filename_res = \
                        current_output_path.format(
                            filename.format("step_{}_results".format(i)))

                    self.plot_reconst(KL, filename_res)

        # Prepare positions for the estimate of
        # eigenvalues to calculate
        positions = []
        for sample in KL.samples:
            positions.append(sample + KL.position)

        evidence.append(self._get_evidence(\
            KL, n_eigs= self._k_indx(positions), eps=1e-3))

        # Plotting final results
        if plot_final:

            filename_res = current_output_path.format(\
                filename.format("posterior_results"))

            self.plot_reconst(KL, filename_res)

        return evidence

    def plot_reconst(self, KL, filename, **kwargs):

        positions = []
        for s in KL.samples:
            positions.append(s + KL.position)

        self._plot_setup(filename, positions, **kwargs)
