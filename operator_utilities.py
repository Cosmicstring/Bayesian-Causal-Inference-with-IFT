import numpy as np

from scipy.stats import norm

import nifty6 as ift
from nifty6 import EnergyOperator, makeOp, SandwichOperator, VdotOperator
from nifty6 import makeDomain, Linearization, FieldAdapter
from nifty6 import Field, MultiField, MultiDomain

def normalize(domain):

    # Note that one would not account for
    # the volume factor here too, since 
    # anyways I am dividing out the volume
    # factor when calculating the cdf
    
    Integrate = ift.ContractionOperator(domain, spaces=None)
    Vdot_t = ift.ContractionOperator(domain, spaces=None)

    return ift.ScalingOperator(domain, 1.0) * \
        Vdot_t.adjoint(Integrate.ptw('reciprocal'))

def rescalemax(domain):

    """
    Rescales the field values such that
    the biggest value is 1.
    """

    # Extract the maximum value, which should be on the last
    # place of the array (likewise for cdf)

    VI_last = ift.ValueInserter(domain, [domain.size-1])
    Vdot_t = ift.ContractionOperator(domain, spaces=None)
    return ift.ScalingOperator(domain, 1.0) * \
        Vdot_t.adjoint(VI_last.adjoint.ptw('reciprocal'))

def findmin(arr):

    """
    Finds index of the minimum of given input field
    """

    return np.where(arr == min(arr))[0]


def findmax(arr):

    return np.where(arr == max(arr))[0]


class myInterpolator(ift.Operator):

    def __init__(self, rg_dom, f_key, z_dom, z_key, shift=False, pieces=1.0, verbose=False,\
        min_z=0., max_z=1.0):
        if not isinstance(rg_dom[0], ift.RGSpace):
            raise TypeError
        if not isinstance(z_dom[0], ift.UnstructuredDomain):
            raise TypeError
        shp = z_dom.shape
        if len(shp) != 1:
            raise ValueError('Z domain shape length incompatible')
        if len(rg_dom.shape) !=1:
            raise ValueError('Interpolator only works in 1D')

        self._domain = ift.MultiDomain.make({f_key: rg_dom,
                                             z_key: z_dom})

        if verbose:
            print("Checking self._z")
            print(rg_dom[0].size)
            print(rg_dom[0].distances[0])

        self._z = np.linspace(min_z, rg_dom[0].distances[0]*rg_dom[0].size,rg_dom[0].size+1)
        self._dist = rg_dom[0].distances[0]
        # I would need RGSpace as target for my f_{X,Y} operator
        self._target = ift.makeDomain(ift.UnstructuredDomain(shp[0]))
        self._f_key = f_key
        self._z_key = z_key
        self._verbose = verbose
        self._shift = shift
        self._pieces = pieces
        
    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        xval = x
        if lin:
            xval = x.val

        # z-coord values, assuming to live on [0,1]
        z_prim = xval[self._z_key]
        # f-field values
        f = xval[self._f_key]

        z_prim = z_prim.val

        if self._shift:
            # This variable is used to shift the z_prim values
            # accordingly to the part of the f_fld which would be
            # left unmasked. In other words, for example:
            #
            # dom of f : [0, 3]
            # unmasked region of f : [1,2]
            # dom of z : [0,1]
            # 
            # Shift would be carried out such that z is  
            # inside the unmasked region of f

            # FIXME: Actually I assume that my z would be
            # anyways inside the [0,1] range, but since it
            # is a latent variable I am free to choose it's
            # domain as it suits me. Maybe there is a way
            # to make this shift in a more general way given
            # domains of f-field values and z-values.

            middle_idx = int(self.domain[self._f_key].size/self._pieces)
            z_prim = z_prim + self._z[middle_idx]

        f = f.val

        # Step size in the rg_dom
        diff = self.domain[self._f_key][0].distances[0]

        i = np.floor(z_prim / diff)

        i = i.astype(int)

        if self._verbose:
            print("TEST indx")
            print(f.shape)
            print('f: ', f)
            print(z_prim.shape)
            print("z_prim: ", z_prim)
            print(diff)
            print("z: ", self._z)
            print("z_prim_at_z: ", self._z[i])
            print("max_i: ", max(i))
            print("min_i: ", min(i))
            

        excess_right = self._z[i] - z_prim
        excess_left = z_prim - self._z[i - 1]

        f_at_z_prim = (f[i - 1] * excess_right + f[i] * excess_left) / diff
        if not lin:
            return ift.Field(self.domain[self._z_key], f_at_z_prim)

        if self._verbose:
            print('f: ', f)
            print('z: ', self._z)
            print('z_prim', z_prim)
            print('i', i)
            print('f at z_prim: ', f_at_z_prim)
            print('excess left: ', excess_left)
            print('excess right: ', excess_right)
            print('diff: ', diff)

        dop_df = np.zeros((2, len(z_prim)))
        dop_df[0, :] = excess_right / diff
        # data[0, np.where(i == 1)] = 0
        dop_df[1, :] = excess_left / diff
        # data[1, np.where(i == len(diff))] = 0

        # FIXME: Not sure why there is this part with
        # multiplying data and self._dist
        
#--->   # data = self._dist*data

        indices = np.zeros((2, len(z_prim)))
        
        # FIXME maybe here is the cause for 
        # boundary bugs

        indices[0, :] = (i - 1) % (self._z.size - 1)
        indices[1, :] = i % (self._z.size - 1)

        if self._verbose:
            print('indices: ', indices)
            print('dop_df:', dop_df)

        dop_dz_prim = (f[i] - f[i-1]) / diff
        ergd1 = SparseOp(self.domain[self._f_key], self.domain[self._z_key],
                         dop_df, np.array(2*[np.arange(len(z_prim)), ]), indices).ducktape(self._f_key)

        ergd2 = ift.makeOp(ift.Field(self.domain[self._z_key], dop_dz_prim)).ducktape(self._z_key)

        if self._verbose:
            print('SPARSE: dop_df : ', dop_df.reshape(-1), '\n',
                  'SPARSE: dom_index : ', np.array(2*[np.arange(len(z_prim)), ]).reshape(-1), '\n',
                  'SPARSE: tar index :', indices.reshape(-1))
        return x.new(ift.Field(self.domain[self._z_key], f_at_z_prim), (ergd1 + ergd2))

#
# Deprecated currently
#

class _VariableCovarianceGaussianEnergy(EnergyOperator):
    """Computes a negative-log Gaussian with unknown covariance.

    Represents up to constants in :math:`s`:

    .. math ::
        E(s,D) = - \\log G(s, D) = 0.5 (s)^\\dagger D^{-1} (s),

    an information energy for a Gaussian distribution with residual s and
    covariance D.

    Parameters
    ----------
    domain : Domain, DomainTuple, tuple of Domain
        Operator domain. By default it is inferred from `s` or
        `covariance` if specified

    residual : key
        residual of the Gaussian. 
    
    inverse_covariance : key
        Inverse covariance of the Gaussian. 

    """

    def __init__(self, domain, dom_residual = None, dom_icov = None, residual_key=None, inverse_covariance_key=None):

        if residual_key == None:
        	raise ValueError("Specify the Gaussian residual field key")
        else:
        	self._residual = residual_key
        
        if inverse_covariance_key == None:
        	raise ValueError("Specify the Gaussian icov key")
        else:
        	self._icov = inverse_covariance_key

        mf_domain = {residual_key : domain, inverse_covariance_key : domain}
        self._domain = ift.MultiDomain.make(mf_domain)

        if dom_residual == None:
        	self._dom_residual = domain
        else:
        	self._dom_residual = dom_residual

        if dom_icov == None:
        	self._dom_icov = domain
        else:
        	self._dom_icov = dom_icov


    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, Linearization)
        xval = x.val if lin else x

        res = .5*xval[self._residual].vdot(xval[self._residual]*xval[self._icov])\
                - .5*xval[self._icov].log().sum()
        if not lin:
            return res

        FA_res = FieldAdapter(self._dom_residual, self._residual)
        FA_sig = FieldAdapter(self._dom_icov, self._icov)
        jac_res = xval[self._residual]*xval[self._icov]
        jac_res = VdotOperator(jac_res)(FA_res)

        # So here we are varying w.r.t. inverse covariance
        jac_sig = .5*(xval[self._residual].absolute()**2)
        jac_sig = VdotOperator(jac_sig)(FA_sig)
        jac_sig = jac_sig - .5*VdotOperator(1./xval[self._icov])(FA_sig)
        jac = (jac_sig + jac_res)(x.jac)

        res = x.new(Field.scalar(res), jac)
        if not x.want_metric:
            return res
        mf = {self._residual:xval[self._icov],
                self._icov:.5*xval[self._icov]**(-2)}
        mf = MultiField.from_dict(mf)
        metric = makeOp(mf)
        metric = SandwichOperator.make(x.jac, metric)
        return res.add_metric(metric)


# Remember, for the confounder model, the f_op lives on a space:
# \hat{f_op} = \hat{f_X} \cross (1,0) + \hat{f_Y} \cross (0,1)
# which gives a space of shape [2*N_data, 1]

class Confounder_merge(ift.LinearOperator):

    def __init__(self, domain_X, key_X, domain_Y, key_Y, target):
        
        dom = {}
        dom[key_X] = domain_X
        dom[key_Y] = domain_Y

        if (domain_X.size + domain_Y.size) != target.size:
            raise ValueError

        self._domain = ift.MultiDomain.make(dom)
        self._target = target
        self._key_X, self._key_Y = key_X, key_Y
        
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._sampling_dtype = np.float64

    def apply(self, x, mode):
        self._check_input(x,mode)

        if mode == self.TIMES:
            fx = x.val[self._key_X]
            fy = x.val[self._key_Y]
            res = np.stack((fx,fy)).flatten()
            return ift.makeField(self._target, res)

        f1 = x.val[:self.domain[self._key_X].size]
        f1 = ift.makeField(self.domain[self._key_X], f1)
        f2 = x.val[self.domain[self._key_Y].size:]
        f2 = ift.makeField(self.domain[self._key_Y], f2)

        res = {self._key_X : f1, self._key_Y : f2}
        res = ift.MultiField.from_dict(res)

        return res        
        
"""
Copyright @ Jakob Roth
"""
class GeomMaskOperator(ift.LinearOperator):
    """
    Takes a field and extracts the central part of the field corresponding to target.shape

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    target : Domain, DomainTuple or tuple of Domain
        The operator's target domain
    """
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        sl = []
        for i in range(len(self._domain.shape)):
            slStart = int((self._domain.shape[i] - self._target.shape[i])/2.)
            slStop = slStart + self._target.shape[i]
            sl.append(slice(slStart, slStop, 1))
        self._slices = tuple(sl)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._slices]
            return ift.Field(self.target, res)
        res = np.zeros(self.domain.shape, x.dtype)
        res[self._slices] = x
        return ift.Field(self.domain, res)

class CmfLinearInterpolator(ift.Operator):
    """
    Parameters
    ----------
    rg_dom : Domain tuple
    point_op : Operator
    """

    def __init__(self, rg_dom, cdf_key, point_dom, point_key, verbose=False):
        if not isinstance(rg_dom[0], ift.RGSpace):
            raise TypeError
        if not isinstance(point_dom[0], ift.UnstructuredDomain):
            raise TypeError
        shp = point_dom.shape
        if len(shp) != 1:
            raise ValueError('Point domain shape length incompatible')
        if len(rg_dom.shape) !=1:
            raise ValueError('CDF interpolator only works in 1D')
        self._domain = ift.MultiDomain.make({cdf_key: rg_dom,
                                             point_key: point_dom})
        self._x = np.linspace(0, rg_dom[0].distances[0]*rg_dom[0].shape[0], rg_dom[0].shape[0] + 1)
        self._dist = rg_dom[0].distances[0]
        # I would need RGSpace as target for my f_{X,Y} operator
        self._target = ift.makeDomain(ift.UnstructuredDomain(shp[0]))
        self._cdf_key = cdf_key
        self._point_key = point_key
        self._verbose = verbose

    def apply(self, x):
        self._check_input(x)
        lin = isinstance(x, ift.Linearization)
        xval = x
        if lin:
            xval = x.val

        # uniform samples
        ps = xval[self._point_key]
        # cdf values
        cdf = xval[self._cdf_key]

        ps = ps.val
        cdf = cdf.val

        # Check whether the samples are in corresponding limits
        if (max(ps)>1.0) or (min(ps)<0) or (min(cdf)<0) or (max(cdf)>1.0):
            print("PS")
            print(ps)
            print("CDF")
            print(cdf)
            raise ValueError
        
        # Can happen sometimes due to numerical roundoff errors
        # that the cdf[-1] < ps_i, for some i, then a problem
        # occurs below. Therefore, setting cdf[-1]=1 should solve
        # the problem. Also, could happen that some of the ps
        # are = 1.0, then again the searchsorted would fail for the
        # last bin, hence one adds a fudge factor 1e-9

        cdf = cdf.copy()
        cdf[-1] = 1.0 + 1e-9

        i = np.searchsorted(cdf, ps, side='left')
        
        if self._verbose:
            print("TEST sorted")
            print(cdf.shape)
            print('cdf: ', cdf)
            print(ps.shape)
            print("ps: ", ps)
            print("i: ", i)

        diff = np.diff(cdf)
        try:
            excess_right = cdf[i] - ps
        except:
            print("Error")
            print("CDF value: {:.5e}".format(cdf[i]))
            print("PS value: {:.5e}".format(ps))

        excess_left = ps - cdf[i - 1]

        x_at_u = (self._x[i - 1] * excess_right + self._x[i] * excess_left) / diff[i - 1]

        if not lin:
            return ift.Field(self.domain[self._point_key], x_at_u)

        if self._verbose:
            print('cdf: ', cdf)
            print('x: ', self._x)
            print('u', ps)
            print('i', i)
            print('x at u: ', x_at_u)
            print('excess left: ', excess_left)
            print('excess right: ', excess_right)
            print('diff: ', diff[i-1])

        data = np.zeros((2, len(ps)))
        data[0, :] = -excess_right / diff[i - 1] ** 2
        # data[0, np.where(i == 1)] = 0
        data[1, :] = -excess_left / diff[i - 1] ** 2
        # data[1, np.where(i == len(diff))] = 0
        data = self._dist*data

        indices = np.zeros((2, len(ps)))
        
        # FIXME maybe here is the cause for 
        # boundary bugs

        indices[0, :] = (i - 1) % len(diff)
        indices[1, :] = i % len(diff)
        if self._verbose:
            print('indices: ', indices)
            print('data:', data)

        dxdu = self._dist / diff[i - 1]
        ergd1 = SparseOp(self.domain[self._cdf_key], self.domain[self._point_key],
                         data, np.array(2*[np.arange(len(ps)), ]), indices).ducktape(self._cdf_key)

        ergd2 = ift.makeOp(ift.Field(self.domain[self._point_key], dxdu)).ducktape(self._point_key)
        if self._verbose:
            print('SPARSE: data : ', data.reshape(-1), '\n',
                  'SPARSE: dom_index : ', np.array(2*[np.arange(len(ps)), ]).reshape(-1), '\n',
                  'SPARSE: tar index :', indices.reshape(-1))
        return x.new(ift.Field(self.domain[self._point_key], x_at_u), (ergd1 + ergd2))


class SparseOp(ift.LinearOperator):
    def __init__(self, dom, tar, arr, dom_index, tar_index):
        from scipy.sparse import coo_matrix
        from scipy.sparse.linalg import aslinearoperator
        self._domain = dom
        self._target = tar
        self._matc = coo_matrix((arr.reshape(-1),
                                (dom_index.reshape(-1), tar_index.reshape(-1))),
                               (tar.size, np.prod(self.domain.shape)))
        self._mat = aslinearoperator(self._matc)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x_val = x.val
        if mode == self.TIMES:
            res = self._mat.matvec(x_val.reshape(-1))
        else:
            res = self._mat.rmatvec(x_val).reshape(self.domain.shape)
        return ift.Field(self._tgt(mode), res)

class IntegrateField(ift.LinearOperator):

    def __init__(self, domain, weight=1, spaces=None):
        self._domain = domain
        self._target = domain
        self._vol = domain[0].distances[0]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        xval = x.val
        if mode == self.TIMES:
            norm_factor = x.integrate().val
            return ift.Field(self.target, np.ones(self.target.shape)*norm_factor)        
        else:
            norm_factor = x.integrate().val
            return ift.Field(self.domain, np.ones(self.domain.shape)*norm_factor)


class NormalizeField(ift.Operator):

    # eht-util for normalization

    def __init__(self, domain):
        self._domain = domain
        self._target = domain

    def apply(self, x):
        self._check_input(x)
        
        if not isinstance(x, ift.Linearization):
            norm_factor = (x.integrate().val)**(-1)
            return ift.Field(self.target, x.val*norm_factor)
        else:
            xfld = x.val
            # .ptw('reciprocal') / .reciprocal
            norm_factor = (xfld.integrate().val)**(-1)
            val = ift.Field(self.target, xfld.val*norm_factor)
            
            # This is not the most optimal way, since the operator
            # would be a full dense matrix, look at page 46 of BCI
            # Hence one can't be doing just 'ift.makeOp'
            jac = ift.Field(self.target,\
                norm_factor - norm_factor**2 * xfld.val)
            jac = ift.makeOp(jac) 
            return x.new(val, jac)

class CDF(ift.LinearOperator):

    """
    Takes in a 'x' (upper limit for cdf integral) and 'pdf' samples. 
    Interpolates the pdf field at points 'x' and the evaluates the 
    integral with 'scipy.integrate.cumtrapz' or 'scipy.integrate.romb'.
    If 'romb' is chosen then 'x.size' = 2^k + 1 samples at which one will
    interpolate the 'pdf' field.

    Input:
    -------

    pdf     :   'pdf' (normalized) field, for which one calculates cdf returned

    Returns:
    --------

    A field representing the cdf for the given pdf
    """

    def __init__(self, domain, verbose=True):
        
        self.verbose = verbose
        self._capability = self.TIMES | self.ADJOINT_TIMES

        if not isinstance(domain[0], ift.RGSpace):
            raise NotImplementedError
        
        # Coordinates at which pdf is defined
        self.xpdf = np.arange(domain.shape[0], dtype=np.float64)*domain[0].distances[0]

        self._domain = domain
        # FIXME: Possibly the same as domain_pdf?

        # I will have the same shape for the output domain
        # as for the input domain, since number of cdf samples
        # i have is equal to the number of samples of the pdf
        # i have.
        self._target = ift.makeDomain(ift.RGSpace(domain.shape[0]+1))
        self._normalize = rescalemax(self._target)

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            xval = x.val
            cdf = np.cumsum(xval)
            
            # FIXME: Possibly exclude this line here, if I append the 0.0 value
            # to the CDF in my CDF operator below?
            cdf = np.append(np.zeros(1), cdf)

            return ift.Field(self.target, cdf)
        else:
            xval = x.val
            # Adjoint to np.cumsum operation
            a_cdf = np.flip(np.cumsum(np.flip(xval)))
            # FIXME: Maybe there is a smarter way than just
            # deleting the last entry here?
            a_cdf = np.delete(a_cdf, -1)
            return ift.Field(self.domain, a_cdf)


class UniformSamples(ift.Operator):

    """
    This operator takes a white gaussian field, Gaussian with
    zero mean and unit_matrix variance. Then, performs a cdf calc 
    and generates a uniformly distributed variable living between [0,1] 
    
    Input:
    ------

    xi   :  ift.Field or ift.MultiField
    key  :  If one has an ift.MultiField then one needs to provde axis
            along which one should integrate

    Returns:
    --------

    Returns an 'ift.Field', which represents CDF values of a white Gaussian
    at the points provided in 'xi'
    """

    def __init__(self, domain, key=''):

        if not isinstance(domain, ift.DomainTuple):
            raise TypeError
        if isinstance(domain, ift.MultiDomain):
            if key=='':
                raise ValueError("Key must be provided"
                                 "works only for 1D case")
            else:
                domain = domain[key]

        self._domain = domain
        self._target = domain

        if key!='':
            self._key = key
        else:
            self._key = ''

    def apply(self, x):
        self._check_input(x)

        # FIXMESeb: Currently works only for 'Fields' and 
        # for 'MultiFields' one needs to provide a key
        # which should be extracted from the multifield
        # in order to perform the integration along that 
        # axis.

        if isinstance(x, ift.Field):
            xval = x.val
            return ift.Field.from_raw(self.target, norm.cdf(xval))
        elif isinstance(x, ift.MultiField):
            if self._key == '':
                raise ValueError("Key must be provided")
            else:
                xval = x.val[self._key]
                return ift.Field.from_raw(self.target, norm.cdf(x, self._pdf, key=self._key))
        elif isinstance(x, ift.Linearization):
            xval = x.val.val
            value = ift.Field.from_raw(self.target, norm.cdf(xval))
            jac = ift.makeOp(ift.Field.from_raw(self.target, norm.pdf(xval)))
            return x.new(value, jac)
