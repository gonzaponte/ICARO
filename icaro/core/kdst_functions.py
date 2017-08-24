import sys
import numpy as np
import tables as tb
import pandas as pd
import invisible_cities.core.fit_functions as fitf
from   invisible_cities.core.system_of_units_c import units
from   invisible_cities.icaro.mpl_functions import set_plot_labels
from   invisible_cities.core.core_functions import in_range
import matplotlib.pyplot as plt
from   collections import namedtuple
import datetime


def delete_lifetime_entry(filename, run_number, delimiter=" ", overwrite=False):
    in_data          = open(filename, "r").readlines()
    header, *in_data = in_data
    out_data         = list(filter(lambda line: int(line.split(delimiter)[0]) != run_number, in_data))

    if len(in_data) == len(out_data): return True
    if overwrite or input("Overwrite value for run {} (y/n)? ".format(run_number)) == "y":
        open(filename, "w").write(header + "".join(out_data))
        return True
    return False


def save_lifetime(    filename, run_number, run_tag,
                            lt,       u_lt,
                           E_0,       u_E0,
                       v_drift,  u_v_drift,
                       t_start,      t_end,            dt,
                    date_start,   date_end,         ddate,
                    comment   = "" ,
                    delimiter = " ",
                    overwrite = False):
    if not delete_lifetime_entry(filename, run_number, overwrite=overwrite):
        return
    line = delimiter.join(map(str, [run_number,    run_tag,        
                                            lt,       u_lt,  
                                           E_0,       u_E0,
                                       v_drift,  u_v_drift,
                                      t_start ,      t_end,    dt, 
                                    date_start,   date_end, ddate,
                                       comment]))

    in_data          = open(filename, "r").readlines()
    in_data.append(line + "\n")

    header, *in_data = in_data
    in_data          = filter(lambda x: not x.isspace(), in_data)
    out_data         = sorted(in_data, key=lambda x: int(x.split(delimiter)[0]))
    open(filename, "w").write(header + "".join(out_data))


def load_lifetimes(filename, delimiter=" ", **kwargs):
    return pd.read_csv(filename, sep=delimiter, **kwargs)


def datetime_to_str(datetime, tformat='%Y-%m-%d-%H:%M:%S'):
    return datetime.strftime(tformat)


def time_from_timestamp(timestamp, tformat='%Y-%m-%d-%H:%M:%S'):
    return datetime.datetime.fromtimestamp(timestamp).strftime(tformat)


def str_to_datetime(timestamp, tformat='%Y-%m-%d-%H:%M:%S'):
    return datetime.datetime.strptime(timestamp, tformat)


def to_deltatime(t0, t1, unit="s", to_str=False):
    delta = pd.Timedelta(t1 - t0, unit=unit)
    return str(delta).replace(" ", "-") if to_str else delta


def lifetime(dst, zrange=(25,530), Erange=(1e+3, 70e3), nbins=10):
    """Compute lifetime as a function of t."""

    print('using data set with length {}'.format(len(dst)))
    st0 = time_from_timestamp(dst.time.values[0])
    st1 = time_from_timestamp(dst.time.values[-1])
    it0 = 0
    it1 = len(dst)
    print('t0 = {} (index = {}) t1 = {} (index = {})'.format(st0, it0, st1, it1))

    indx  = int(len(dst) / nbins)
    print('bin length = {}'.format(indx))

    CHI2    = []
    LAMBDA  = []
    ELAMBDA = []
    TSTAMP  = []

    for i in range(nbins):
        k0 = i * indx
        k = (i+1) * indx - 1
        print(' ---fit over events between {} and {}'.format(k0, k))
        st0 = time_from_timestamp(dst.time.values[k0])
        st =  time_from_timestamp(dst.time.values[k])

        print('time0 = {} time1 = {}'.format(st0,st))

        tleg = dst[in_range(dst.time.values, minval=dst.time.values[k0], maxval=dst.time.values[k])]
        print('size of time leg = {}'.format(len(tleg)))
        F, x, y, sy = profile_and_fit(tleg.Z, tleg.S2e,
                                      xrange=zrange,
                                      yrange=Erange,
                                      nbins=nbins,
                                      fitpar=(50000,-300),
                                      label=("Drift time ($\mu$s)", "S2 energy (pes)"))
        print_fit(F)
        chi = chi2(F, x, y, sy)
        print('chi2 = {}'.format(chi))
        CHI2.append(chi)
        LAMBDA.append(F.values[1])
        ELAMBDA.append(F.errors[1])
        TSTAMP.append(st)

    TIME = [datetime.datetime.strptime(elem,
           '%Y-%m-%d %H:%M:%S') for elem in TSTAMP]

    return CHI2, LAMBDA, ELAMBDA, TSTAMP, TIME


def lifetime_vs_t(dst, nslices=10, nbins=10, seed=(3e4, -5e2), timestamps=None, **profOpt):
    LT, LTu = [], []
    E0, E0u = [], []
    T ,  Tu = [], []

    tmin = np.min(dst.time)
    tmax = np.max(dst.time)
    bins = np.linspace(tmin, tmax, nslices+1)
    for t0, t1 in zip(bins[:-1], bins[1:]):
        subdst   = dst[in_range(dst.time, t0, t1)]
        Z, E, Eu = fitf.profileX(subdst.Z, subdst.S2e, nbins, **profOpt)
        f        = fitf.fit(fitf.expo, Z, E, seed, sigma=Eu)

        LT .append(-f.values[1] )
        LTu.append( f.errors[1] )
        E0 .append(f.values[0] )
        E0u.append( f.errors[0] )
        T  .append(0.5*(t1 + t0))
        Tu .append(0.5*(t1 - t0))

    plt.errorbar(T, LT, LTu, Tu)

    #plt.errorbar(T, E0, E0u, Tu)
    return T, LT, Tu, LTu, E0, E0u

class MapXY:
    def __init__(self, x, y, E):
        self.xs   = x.reshape(x.size, 1) #file to column vector
        self.ys   = y.reshape(y.size, 1)
        self.eref = E[E.shape[0]//2, E.shape[1]//2]
        self.es   = E
        print('reference energy = {}'.format(self.eref))

    def xycorr(self, x, y):
        x_closest  = np.apply_along_axis(np.argmin, 0, abs(x - self.xs))
        y_closest  = np.apply_along_axis(np.argmin, 0, abs(y - self.ys))
        e          = self.es[x_closest, y_closest]
        e[e < 1e3] = self.eref
        return self.eref / e


def load_dst(filename):
    with tb.open_file(filename) as h5:
        return pd.DataFrame.from_records(h5.root.DST.Events.read())


def event_rate(kdst):
    t0 = np.min(kdst.time)
    t1 = np.max(kdst.time)
    return kdst.event.size/(t1-t0)


def profile_and_fit(X, Y, xrange, yrange, nbins, fitpar, label, fitOpt  = "r"):
    n_it = 0
    chi2 = 0
    #while not 0.8 < chi2 < 1.4:
    xe       = 0.5*(xrange[1] - xrange[0])/nbins
    x, y, sy = fitf.profileX(X, Y, nbins=nbins, xrange=xrange, yrange=yrange)
    sel      = fitf.in_range(x, *xrange)
    x, y, sy = x[sel], y[sel], sy[sel]
    f        = fitf.fit(fitf.expo, x, y, fitpar, sigma=sy)
    chi2     = f.chi2
    nbins   -= n_it

     #   n_it += 1
      #  if nbins < 5:
       #     print("Chi2 does not get close to 1")
        #    break

    plt.errorbar(x=x, xerr=xe, y=y, yerr=sy,
                 linestyle='none', marker='.')
    plt.plot(x, f.fn(x), fitOpt)
    set_plot_labels(xlabel=label[0], ylabel=label[1], grid=True)
    return f, x, y, sy


def profile_and_fit_radial(X, Y, xrange, yrange, nbins, fitpar, label, fitOpt="r"):
    xe       = (xrange[1] - xrange[0])/nbins
    x, y, sy = fitf.profileX(X, Y, nbins=nbins, xrange=xrange, yrange=yrange)
    sel      = fitf.in_range(x, *xrange)
    x, y, sy = x[sel], y[sel], sy[sel]
    f        = fitf.fit(fitf.polynom, x, y, fitpar, sigma=sy)

    plt.errorbar(x=x, xerr=xe, y=y, yerr=sy,
                 linestyle='none', marker='.')
    plt.plot(x, f.fn(x), fitOpt)
    set_plot_labels(xlabel=label[0], ylabel=label[1], grid=True)
    return f, x, y, sy


def chi2(F, X, Y, SY):
    fitx = F.fn(X)
    n = len(F.values)
    print('degrees of freedom = {}'.format(n))
    chi2t = 0
    for i, x in enumerate(X):
        chi2 = abs(Y[i] - fitx[i])/SY[i]
        chi2t += chi2
        #print('x = {} f(x) = {} y = {} ey = {} chi2 = {}'.format(
               #x, fitx[i], Y[i], SY[i], chi2 ))
    return chi2t/(len(X)-n)

    #chi2 = np.sum(np.ma.masked_invalid((fitx - y)**2/sy**2))
    #print('chi2 = {}'.format(chi2))

def print_fit(f):
    for i, val in enumerate(f.values):
        print('fit par[{}] = {} error = {}'.format(i, val, f.errors[i]))
