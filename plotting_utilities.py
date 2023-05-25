import nifty6 as ift
from nifty6 import makeDomain
from nifty6 import Field
from nifty6.domains.gl_space import GLSpace
from nifty6.domains.hp_space import HPSpace
from nifty6.domains.power_space import PowerSpace
from nifty6.domains.rg_space import RGSpace
from nifty6.domains.unstructured_domain import UnstructuredDomain

from nifty6.field import Field
from nifty6.minimization.iteration_controllers import EnergyHistory

import numpy as np
import matplotlib.pyplot as plt
import os


def _makeplot(name, block=True, dpi=None):
    import matplotlib.pyplot as plt
    if name is None:
        plt.show(block=block)
        if block:
            plt.close()
        return
    extension = os.path.splitext(name)[1]
    if extension in (".pdf", ".png", ".svg"):
        args = {}
        if dpi is not None:
            args['dpi'] = float(dpi)
        plt.savefig(name, **args)
        plt.close()
    else:
        raise ValueError("file format not understood")


def _limit_xy(**kwargs):
    import matplotlib.pyplot as plt
    x1 = kwargs.pop("xmin", None)
    x2 = kwargs.pop("xmax", None)
    y1 = kwargs.pop("ymin", None)
    y2 = kwargs.pop("ymax", None)

    xbool = not(isinstance(x1, type(None))) and not(isinstance(x2, type(None)))
    ybool = not(isinstance(y1, type(None))) and not(isinstance(y2, type(None)))

    if xbool and ybool:
        plt.axis((x1, x2, y1, y2))

def _register_cmaps():
    try:
        if _register_cmaps._cmaps_registered:
            return
    except AttributeError:
        _register_cmaps._cmaps_registered = True

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    planckcmap = {'red':   ((0., 0., 0.), (.4, 0., 0.), (.5, 1., 1.),
                            (.7, 1., 1.), (.8, .83, .83), (.9, .67, .67),
                            (1., .5, .5)),
                  'green': ((0., 0., 0.), (.2, 0., 0.), (.3, .3, .3),
                            (.4, .7, .7), (.5, 1., 1.), (.6, .7, .7),
                            (.7, .3, .3), (.8, 0., 0.), (1., 0., 0.)),
                  'blue':  ((0., .5, .5), (.1, .67, .67), (.2, .83, .83),
                            (.3, 1., 1.), (.5, 1., 1.), (.6, 0., 0.),
                            (1., 0., 0.))}
    he_cmap = {'red':   ((0., 0., 0.), (.167, 0., 0.), (.333, .5, .5),
                         (.5, 1., 1.), (1., 1., 1.)),
               'green': ((0., 0., 0.), (.5, 0., 0.), (.667, .5, .5),
                         (.833, 1., 1.), (1., 1., 1.)),
               'blue':  ((0., 0., 0.), (.167, 1., 1.), (.333, .5, .5),
                         (.5, 0., 0.), (1., 1., 1.))}
    fd_cmap = {'red':   ((0., .35, .35), (.1, .4, .4), (.2, .25, .25),
                         (.41, .47, .47), (.5, .8, .8), (.56, .96, .96),
                         (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                         (.9, .5, .5), (1., .4, .4)),
               'green': ((0., 0., 0.), (.2, 0., 0.), (.362, .88, .88),
                         (.5, 1., 1.), (.638, .88, .88), (.8, .25, .25),
                         (.9, .3, .3), (1., .2, .2)),
               'blue':  ((0., .35, .35), (.1, .4, .4), (.2, .8, .8),
                         (.26, .8, .8), (.41, 1., 1.), (.44, .96, .96),
                         (.5, .8, .8), (.59, .47, .47), (.8, 0., 0.),
                         (1., 0., 0.))}
    fdu_cmap = {'red':   ((0., 1., 1.), (0.1, .8, .8), (.2, .65, .65),
                          (.41, .6, .6), (.5, .7, .7), (.56, .96, .96),
                          (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                          (.9, .5, .5), (1., .4, .4)),
                'green': ((0., .9, .9), (.362, .95, .95), (.5, 1., 1.),
                          (.638, .88, .88), (.8, .25, .25), (.9, .3, .3),
                          (1., .2, .2)),
                'blue':  ((0., 1., 1.), (.1, .8, .8), (.2, 1., 1.),
                          (.41, 1., 1.), (.44, .96, .96), (.5, .7, .7),
                          (.59, .42, .42), (.8, 0., 0.), (1., 0., 0.))}
    pm_cmap = {'red':   ((0., 1., 1.), (.1, .96, .96), (.2, .84, .84),
                         (.3, .64, .64), (.4, .36, .36), (.5, 0., 0.),
                         (1., 0., 0.)),
               'green': ((0., .5, .5), (.1, .32, .32), (.2, .18, .18),
                         (.3, .8, .8),  (.4, .2, .2), (.5, 0., 0.),
                         (.6, .2, .2), (.7, .8, .8), (.8, .18, .18),
                         (.9, .32, .32), (1., .5, .5)),
               'blue':  ((0., 0., 0.), (.5, 0., 0.), (.6, .36, .36),
                         (.7, .64, .64), (.8, .84, .84), (.9, .96, .96),
                         (1., 1., 1.))}

    plt.register_cmap(cmap=LinearSegmentedColormap("Planck-like", planckcmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("High Energy", he_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Map", fd_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Uncertainty",
                                                   fdu_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Plus Minus", pm_cmap))

def _plot2D(f, ax, **kwargs):
    import matplotlib.pyplot as plt

    dom = f.domain

    if len(dom) > 2:
        raise ValueError("DomainTuple can have at most two entries.")

    # check for multifrequency plotting
    have_rgb = False
    x_space = 0
    if len(dom) == 2:
        f_space = kwargs.pop("freq_space_idx", 1)
        if f_space not in [0, 1]:
            raise ValueError("Invalid frequency space index")
        if (not isinstance(dom[f_space], RGSpace)) \
           or len(dom[f_space].shape) != 1:
            raise TypeError("Need 1D RGSpace as frequency space domain")
        x_space = 1 - f_space

        # Only one frequency?
        if dom[f_space].shape[0] == 1:
            from .sugar import makeField
            f = makeField(f.domain[x_space],
                          f.val.squeeze(axis=dom.axes[f_space]))
        else:
            val = f.val
            if f_space == 0:
                val = np.moveaxis(val, 0, -1)
            rgb = _rgb_data(val)
            have_rgb = True

    foo = kwargs.pop("norm", None)
    norm = {} if foo is None else {'norm': foo}

    foo = kwargs.pop("aspect", None)
    aspect = {} if foo is None else {'aspect': foo}

    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))
    dom = dom[x_space]
    if not have_rgb:
        cmap = kwargs.pop("cmap", plt.rcParams['image.cmap'])

    if isinstance(dom, RGSpace):
        nx, ny = dom.shape
        dx, dy = dom.distances
        if have_rgb:
            im = ax.imshow(
                rgb, extent=[0, nx*dx, 0, ny*dy], origin="lower", **norm,
                **aspect)
        else:
            im = ax.imshow(
                f.val.T, extent=[0, nx*dx, 0, ny*dy],
                vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"),
                cmap=cmap, origin="lower", **norm, **aspect)
            plt.colorbar(im)
        _limit_xy(**kwargs)
        return
    elif isinstance(dom, (HPSpace, GLSpace)):
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)
        if have_rgb:
            res = np.full(shape=res.shape+(3,), fill_value=1.,
                          dtype=np.float64)

        if isinstance(dom, HPSpace):
            ptg = np.empty((phi.size, 2), dtype=np.float64)
            ptg[:, 0] = theta
            ptg[:, 1] = phi
            base = pyHealpix.Healpix_Base(int(np.sqrt(dom.size//12)), "RING")
            if have_rgb:
                res[mask] = rgb[base.ang2pix(ptg)]
            else:
                res[mask] = f.val[base.ang2pix(ptg)]
        else:
            ra = np.linspace(0, 2*np.pi, dom.nlon+1)
            dec = pyHealpix.GL_thetas(dom.nlat)
            ilat = _find_closest(dec, theta)
            ilon = _find_closest(ra, phi)
            ilon = np.where(ilon == dom.nlon, 0, ilon)
            if have_rgb:
                res[mask] = rgb[ilat*dom[0].nlon + ilon]
            else:
                res[mask] = f.val[ilat*dom.nlon + ilon]
        plt.axis('off')
        if have_rgb:
            plt.imshow(res, origin="lower")
        else:
            plt.imshow(res, vmin=kwargs.get("vmin"), vmax=kwargs.get("vmax"),
                       norm=norm.get('norm'), cmap=cmap, origin="lower")
            plt.colorbar(orientation="horizontal")
        return
    raise ValueError("Field type not(yet) supported")


def _plot(f, ax, xcoords, **kwargs):
    _register_cmaps()
    if isinstance(f, Field) or isinstance(f, EnergyHistory):
        f = [f]
    f = list(f)
    if len(f) == 0:
        raise ValueError("need something to plot")
    if isinstance(f[0], EnergyHistory):
        _plot_history(f, ax, **kwargs)
        return
    if not isinstance(f[0], Field):
        raise TypeError("incorrect data type")
    dom1 = f[0].domain
    if (len(dom1) == 1 and
        (isinstance(dom1[0], PowerSpace) or
         (isinstance(dom1[0], RGSpace) or 
            (isinstance(dom1[0], UnstructuredDomain)) and
          len(dom1[0].shape) == 1))):

        if isinstance(dom1[0], UnstructuredDomain) and \
            isinstance(xcoords, type(None)):
                raise ValueError 

        _plot1D(f, ax, xcoords, **kwargs)
        return
    else:
        if len(f) != 1:
            raise ValueError("need exactly one Field for 2D plot")
        _plot2D(f[0], ax, **kwargs)
        return
    raise ValueError("Field type not(yet) supported")

def _plot1D(f, ax, xcoord_f, **kwargs):

    _register_cmaps()

    if isinstance(f, Field):
        f = [f]
    f = list(f)
    if len(f) == 0:
        raise ValueError("need something to plot")
    if not isinstance(f[0], Field):
        raise TypeError("incorrect data type")

    scatter_flags = kwargs.pop("scatter", False)
    if not isinstance(scatter_flags, list):
        scatter_flags = [scatter_flags] * len(f)

    marker = kwargs.pop("marker", 'o')
    if not isinstance(marker, list):
        marker = [marker] * len(f)

    label = kwargs.pop("label", None)
    if not isinstance(label, list):
        label = [label] * len(f)

    linewidth = kwargs.pop("linewidth", 1.)
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * len(f)

    alpha = kwargs.pop("alpha", None)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(f)

    color = kwargs.pop("color", None)
    if not isinstance(color, list):
        color = [color] * len(f)

    _sorted = kwargs.pop("sorted", False)

    if not isinstance(_sorted, list):
        _sorted = [_sorted] * len(f)

    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))

    secondary = kwargs.pop("secondary", False)

    if secondary:
        ax2 = ax.twinx()
        ax2.tick_params(axis='y')
        axes = [ax, ax2]

        if len(axes) != len(f):
            raise ValueError("Lengths for 'secondary' don't match")

    for i, fld in enumerate(f):
        dom = fld.domain[0]
        if isinstance(dom, RGSpace):
            
            npoints = dom.shape[0]
            dist = dom.distances[0]
                
            xmin = kwargs.get("xmin", None)
            xmax = kwargs.get("xmax", None)
            
            ycoord = fld.val

            if not(isinstance(xmin, type(None))) and not(isinstance(xmax, type(None))):
                xcoord = np.arange(xmin,xmax, (xmax-xmin)/npoints, dtype=np.float64)
                # Since it can happen that np.arange adds one more point after 'stop'
                # I check here whether this new 'xcoord' agrees in size and then cut
                # out the last point if it is there
                if xcoord.size == ycoord.size:
                    pass 
                elif (xcoord.size - ycoord.size)==1:
                    xcoord = xcoord[:-1]
                else:
                    raise ValueError("New 'xcoord' size and 'ycoord' size don't agree")
            else:
                xcoord = np.arange(npoints, dtype=np.float64)*dist

            if secondary == True:
                # axes[i].set_yscale(kwargs.pop("yscale","linear"))
                axes[i].plot(xcoord, ycoord, label=label[i], color=color[i],\
                    linewidth=linewidth[i], alpha=alpha[i])
            else:
                plt.yscale(kwargs.pop("yscale", "linear"))
                plt.plot(xcoord, ycoord, label=label[i], color=color[i],\
                    linewidth=linewidth[i], alpha=alpha[i])
            
            _limit_xy(**kwargs)
            if label != ([None]*len(f)):
                plt.legend()
        elif isinstance(dom, PowerSpace):
            plt.xscale(kwargs.pop("xscale", "log"))
            plt.yscale(kwargs.pop("yscale", "log"))
            xcoord = dom.k_lengths
            ycoord = fld.val_rw()
            ycoord[0] = ycoord[1]
            plt.plot(xcoord, ycoord, label=label[i], color=color[i], \
                linewidth=linewidth[i], alpha=alpha[i])
            
            # Custom tweaks
            plt.ylabel(r'$p(| k |)$', fontsize=24)
            plt.xlabel(r'$| k |$', fontsize=24)

            plt.yticks(fontsize=18)
            plt.xticks(fontsize=18)

            plt.rcParams.update({'font.size': 22})

            _limit_xy(**kwargs)
            if label != ([None]*len(f)):
                plt.legend()
        elif (isinstance(dom, ift.UnstructuredDomain)) or (scatter_flags[i]): 

            if len(scatter_flags) != len(f):
                raise ValueError("scatter_flags param and Field list don't match in length!")

            if not(isinstance(xcoord_f, list)):
                xcoord_f = [xcoord_f]; xcoord_f = list(xcoord_f)

            ycoord = f[i].val
            xcoord = xcoord_f[i]

            if not _sorted[i]:
                indx = xcoord.argsort()
            
            if secondary == True:
                
                # axes[i].set_yscale(kwargs.pop("yscale", "linear"))
                # axes[i].set_xscale(kwargs.pop("xscale", "linear"))

                if scatter_flags[i]:
                    axes[i].scatter(xcoord, ycoord, label=label[i], color = color[i],\
                        linewidth=linewidth[i], alpha=alpha[i], marker = marker[i])
                else:
                    axes[i].plot(xcoord[indx], ycoord[indx], label=label[i], color = color[i], \
                        linewidth=linewidth[i], alpha = alpha[i])

            else:   
                plt.yscale(kwargs.pop("yscale", "linear"))
                plt.xscale(kwargs.pop("xscale", "linear"))
                # FIXME One should think here about whether a scatter plot
                # of a shape which has more than 1 axes should be done this way
                if scatter_flags[i]:
                    plt.scatter(xcoord, ycoord, label=label[i], color=color[i],\
                        linewidth=linewidth[i], alpha=alpha[i], marker = marker[i])
                else:
                    if not _sorted[i]:
                        plt.plot(xcoord[indx], ycoord[indx], label=label[i], color=color[i],\
                            linewidth=linewidth[i], alpha = alpha[i])
                    else:
                        plt.plot(xcoord, ycoord, label=label[i], color=color[i],\
                            linewidth=linewidth[i], alpha = alpha[i])
            _limit_xy(**kwargs)

            if label != ([None]*len(f)):
                plt.legend()
        else:
            raise ValueError("Plotting routine for this not implemented")


class myPlot(ift.Plot):
    """
    If the domains are included in ift.Plot, then it just forwards the fields to
    ift.Plot, otherwise builds up RGSpace to plot the field with ift.Plot
    """

    def __init__(self):
        self._plotts = []
        self._kwarrgs = []
        self._xcoords = []
        self.ift_plot = ift.Plot()

    def my_add(self, f, **kwargs):
        
        xcoord = kwargs.pop("xcoord", [])
        self._xcoords.append(xcoord)

        self._plotts.append(f)
        self._kwarrgs.append(kwargs)

    def my_output(self, **kwargs):
        nplot = len(self._plotts)
        fig = plt.figure()
        if "title" in kwargs:
            plt.suptitle(kwargs.pop("title"))
        nx = kwargs.pop("nx", 0)
        ny = kwargs.pop("ny", 0)
        if nx == ny == 0:
            nx = ny = int(np.ceil(np.sqrt(nplot)))
        elif nx == 0:
            nx = np.ceil(nplot/ny)
        elif ny == 0:
            ny = np.ceil(nplot/nx)
        if nx*ny < nplot:
            raise ValueError(
                'Figure dimensions not sufficient for number of plots. '
                'Available plot slots: {}, number of plots: {}'
                .format(nx*ny, nplot))
        xsize = kwargs.pop("xsize", 6)
        ysize = kwargs.pop("ysize", 6)
        fig.set_size_inches(xsize, ysize)

        for i in range(nplot):
            ax = fig.add_subplot(ny, nx, i+1)
            _plot(self._plotts[i], ax, self._xcoords[i], \
                **self._kwarrgs[i])
        fig.tight_layout()
        _makeplot(kwargs.pop("name", None), block=kwargs.pop(
            "block", True), dpi=kwargs.pop("dpi", None))
