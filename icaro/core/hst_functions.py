import textwrap
import functools

import numpy             as np
import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm

import invisible_cities.core.fit_functions as fitf

def labels(xlabel, ylabel, title=""):
    """
    Set x and y labels.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title ( title)


def shift_to_bin_centers(x):
    """
    Return bin centers, given bin lower edges.
    """
    return x[:-1] + np.diff(x) * 0.5


def hist(*args, **kwargs):
    """
    Create a figure and then the histogram
    """
    if kwargs.get("new_figure", True):
        plt.figure()
    y, x, p = plt.hist(*args, **kwargs)
    return y, shift_to_bin_centers(x), p


def doublehist(data1, data2, lbls, *args, **kwargs):
    """
    Create a figure and then the histogram
    """
    h1 = hist(data1, *args, label=lbls[0], alpha=0.5, normed=True, new_figure=True, **kwargs)
    h2 = hist(data2, *args, label=lbls[1], alpha=0.5, normed=True, new_figure=False, **kwargs)
    return h1, h2, plt.legend()


def hist2d(*args, **kwargs):
    """
    Create a figure and then the histogram
    """
    if kwargs.get("new_figure", True):
        plt.figure()
    z, x, y, p = plt.hist2d(*args, **kwargs)
    return z, shift_to_bin_centers(x), shift_to_bin_centers(y), p


def pdf(data, *args, **kwargs):
    """
    Create a figure and then the normalized histogram
    """
    if kwargs.get("new_figure", True):
        plt.figure()
    h = hist(data, *args, **kwargs, weights=np.ones_like(data)/len(data))
    plt.yscale("log")
    plt.ylim(1e-4, 1.)
    return h


def scatter(*args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    if kwargs.get("new_figure", True):
        plt.figure()
    return plt.scatter(*args, **kwargs)


# I will leave this function here so old code does not crash,
# but the user will want to use the one after that
def profile_and_scatter(x, y, z, nbin, *args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    if kwargs.get("new_figure", True):
        plt.figure()
    x, y, z, ze = fitf.profileXY(x, y, z, *nbin, *args, **kwargs)
    x_ = np.repeat(x, x.size)
    y_ = np.tile  (y, y.size)
    z_ = z.flatten()
    return (x, y, z, ze), plt.scatter(x_, y_, c=z_, marker="s"), plt.colorbar()


def hist2d_profile(x, y, z, nbinx, nbiny, xrange, yrange, **kwargs):
    """
    Create a profile 2d of the data and plot it as an histogram.
    """
    plt.figure()
    x, y, z, ze = fitf.profileXY(x, y, z, nbinx, nbiny, xrange, yrange)
    x_ = np.repeat(x, x.size)
    y_ = np.tile  (y, y.size)
    z_ = z.flatten()
    h  = hist2d(x_, y_, (nbinx, nbiny), (xrange, yrange), weights=z_, **kwargs)
    return (x, y, z, ze), h, plt.colorbar()


def doublescatter(x1, y1, x2, y2, lbls, *args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    if kwargs.get("new_figure", True):
        plt.figure()
    sc1 = scatter(x1, y1, *args, label=lbls[0], **kwargs)
    sc2 = scatter(x2, y2, *args, label=lbls[1], new_figure=False, **kwargs)
    return sc1, sc2, plt.legend()


def covariance(x, y):
    cov = np.cov(x, y)
    l, v = np.linalg.eig(cov)
    lx, ly = l**0.5
    vx, vy = v.T
    x0, y0 = np.mean(x), np.mean(y)
    x1     = lx * vx[0]
    y1     = lx * vx[1]
    plt.arrow(x0, y0, x1, y1, head_width=0.1*ly, head_length=0.1*lx, fc='r', ec='r')
    x1     = ly * vy[0]
    y1     = ly * vy[1]
    plt.arrow(x0, y0, x1, y1, head_width=0.1*lx, head_length=0.1*ly, fc='r', ec='r')
    return l, v


def resolution(values, E_from=41.5, E_to=2458):
    amp, mu, sigma, *_ = values
    r = 235. * sigma/mu
    return r, r * (E_from/E_to)**0.5


def gausstext(values, E_from=41.5, E_to=2458):
    return textwrap.dedent("""
        $\mu$ = {0:.1f}
        $\sigma$ = {1:.2f}
        R = {2:.3}% @ {4} keV
        Rbb = {3:.3}% @ {5}""".format(*values[1:3], *resolution(values),
                                      E_from, "Qbb" if E_to==2458 else str(E_to) + " keV"))


def save_to_folder(outputfolder, name):
    """
    Set title and save plot in folder.
    """
    plt.savefig("{}/{}.png".format(outputfolder, name), dpi=100)
