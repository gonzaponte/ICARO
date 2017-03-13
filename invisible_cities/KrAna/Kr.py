"""
Kr analysis
"""
from __future__ import print_function, division

import os
import functools
import time
import glob
print("Running on ", time.asctime())

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"]          = 12, 10
plt.rcParams["font.size"]               = 15
plt.rcParams["figure.max_open_warning"] = 100

import invisible_cities.database.load_db as DB
import invisible_cities.core.system_of_units_c as SystemOfUnits
import invisible_cities.reco.pmaps_functions as pmapf
import invisible_cities.core.fit_functions as fitf

DataPMT = DB.DataPMT()
DataSiPM = DB.DataSiPM()
units = SystemOfUnits.SystemOfUnits()
pi = np.pi

def width(times, to_mus=False):
    w = (np.max(times) - np.min(times)) if np.any(times) else 0.
    return w * units.ns/units.mus if to_mus else w


def timefunc(f):
    def time_f(*args, **kwargs):
        t0 = time.time()
        output = f(*args, **kwargs)
        print("Time spent in {}: {} s".format(f.__name__,
                                              time.time() - t0))
        return output
    return time_f


def integrate_charge(d):
    newd = dict((key, np.sum(value)) for key, value in d.items())
    return map(np.array, list(zip(*newd.items())))

profOpt = "--k"
fitOpt  = "r"


class Event:
    """
    Store for relevant event info.
    """
    def __init__(self):
        self.nS1   = 0
        self.S1w   = []
        self.S1h   = []
        self.S1i   = []

        self.nS2   = 0
        self.S2w   = []
        self.S2h   = []
        self.S2i   = []

        self.Nsipm = 0
        self.Q     = 0
        self.Xt    = np.nan
        self.Yt    = np.nan
        self.X     = np.nan
        self.Y     = np.nan
        self.Xrms  = np.nan
        self.Yrms  = np.nan
        self.Z     = np.nan
        self.R     = np.nan
        self.Phi   = np.nan

        self.ok    = False


class Dataset:
    """
    Trick for accesing event properties as an attribute of the dataset.
    """
    @timefunc
    def __init__(self, evts, mask=False):
        self.evts = np.array(evts, dtype=object)
        if mask:
            self.evts = self.evts[np.array([evt.ok for evt in evts])]

        for attr in filter(lambda x: not x.endswith("__"), Event().__dict__):
            x = []
            for evt in self.evts:
                a = getattr(evt, attr)
                if hasattr(a, "__iter__"):
                    x.extend(a)
                else:
                    x.append(a)
            setattr(self, attr, np.array(x))

    def apply_mask(self, mask):
        for attr in filter(lambda x: not x.endswith("__"), self.__dict__):
            setattr(self, attr, getattr(self, attr)[mask])


@timefunc
def fill_info(s1df, s2df, sidf, evts_out, ifile=None):
    s1s = pmapf.df_to_pmaps_dict(s1df)
    s2s = pmapf.df_to_pmaps_dict(s2df)
    sis = pmapf.df_to_s2si_dict (sidf)
    
    evts = set(list(s1s.keys()) +
               list(s2s.keys()) +
               list(sis.keys()))
    nevt = len(evts)
    print(ifile, nevt)
    for i, evt_ in enumerate(evts):
        evt = Event()
        s1  = s1s.get(evt_, dict())
        s2  = s2s.get(evt_, dict())
        si  = sis.get(evt_, dict())
        
        evt.nS1 = len(s1)
        evt.nS2 = len(s2)

        s1time = 0
        for peak, (t, e) in s1.items():
            try:
                evt.S1w.append(width(t))
                evt.S1h.append(np.max(e))
                evt.S1i.append(np.sum(e))
                s1time = t[np.argmax(e)]
            except:
                print(evt_, peak, t, e)

        s2time = 0
        for peak, (t, e) in s2.items():
            evt.S2w.append(width(t, to_mus=True))
            evt.S2h.append(np.max(e))
            evt.S2i.append(np.sum(e))
            s2time = t[np.argmax(e)]

            if not peak in si:
                continue
            IDs, Qs = integrate_charge(si[peak])
            Qtot    = np.sum(Qs)
            xsipms  = DataSiPM.X.values[IDs]
            ysipms  = DataSiPM.Y.values[IDs]

            evt.Nsipm = len(IDs)
            evt.Q     = Qtot
            evt.X     = np.average(xsipms, weights=Qs)
            evt.Y     = np.average(ysipms, weights=Qs)
            evt.Xrms  = np.sum(Qs * (xsipms-evt.X)**2) / (Qtot - 1)
            evt.Yrms  = np.sum(Qs * (ysipms-evt.Y)**2) / (Qtot - 1)
            evt.R     = (evt.X**2 + evt.Y**2)**0.5
            evt.Phi   = np.arctan2(evt.Y, evt.X)

        evt.ok = evt.nS1 == evt.nS2 == 1
        if evt.ok:
            evt.Z = (s2time - s1time) * units.ns / units.mus
        evts_out.append(evt)


@timefunc
def fill_events(inputfiles):
    evts_out = []
    for ifile in inputfiles:
        s1s, s2s, sis = pmapf.read_pmaps(ifile)
        fill_info(s1s, s2s, sis, evts_out, ifile)
    return evts_out


def labels(xlabel, ylabel):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def pdf(data, *args, **kwargs):
    data = np.array(data)
    plt.figure()
    plt.hist(data, *args, **kwargs, weights=np.ones_like(data)/len(data))
    plt.yscale("log")
    plt.ylim(1e-4, 1.)


def save_to_folder(outputfolder, name):
    plt.title(name)
    plt.savefig("{}/{}.png".format(outputfolder, name), dpi=100)


@timefunc
def plot_S12_info(data, outputfolder="plots/"):
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    save = functools.partial(save_to_folder, outputfolder)

    ################################
    pdf(data.nS1, 5, range=(0, 5))
    labels("# S1", "Entries")
    save("NS1")
    
    ################################
    pdf(data.nS2, 5, range=(0, 5))
    labels("# S2", "Entries")
    save("NS2")
    
    ################################
    pdf(data.S1w, 20, range=(0, 500))
    labels("S1 width (ns)", "Entries")
    save("S1width")
    
    ################################
    pdf(data.S2w, 50, range=(0, 30))
    labels("S2 width ($\mu$s)", "Entries")
    save("S2width")
    
    ################################
    pdf(data.S1h, 40, range=(0, 8))
    labels("S1 height (pes)", "Entries")
    save("S1height")
    
    ################################
    pdf(data.S2h, 50, range=(0, 5e3))
    labels("S2 height (pes)", "Entries")
    save("S2height")
    
    ################################
    pdf(data.S1i, 50, range=(0, 50))
    labels("S1 integral (pes)", "Entries")
    save("S1integral")
    
    ################################
    pdf(data.S2i, 50, range=(0, 8e3))
    labels("S2 integral (pes)", "Entries")
    save("S2integral")


@timefunc
def plot_evt_info(data, outputfolder="plots/"):
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    save = functools.partial(save_to_folder, outputfolder)
    
    ################################
    plt.figure()
    plt.hist(data.S1i, 40, range=(0, 20))
    labels("S1 energy (pes)", "Entries")
    save("S1spectrum")
    
    ################################
    plt.figure()
    plt.hist(data.S2i, 50, range=(3e3, 8e3))
    labels("S2 energy (pes)", "Entries")
    save("S2spectrum")
    
    ################################
    pdf(data.S1i, 40, range=(0, 20))
    labels("S1 energy (pes)", "Entries")
    save("S1spectrum_log")
    
    ################################
    pdf(data.S2i, 50, range=(3e3, 8e3))
    labels("S2 energy (pes)", "Entries")
    save("S2spectrum_log")
    
    ################################
    plt.figure()
    plt.hist(data.Z, 100)
    labels("Drift time ($\mu$s)", "Event energy (pes)")
    save("Z")
    
    ################################
    plt.figure()
    plt.scatter(data.Z, data.S2i)
    x, y, _ = fitf.profileX(data.Z, data.S2i, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.expo, x, y, (7e3, -1))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 4200, "{:.1f} $\cdot$ exp(-x/{:.4g})".format(*f.values))
    labels("Drift time ($\mu$s)", "Event energy (pes)")
    plt.ylim(4e3, 8e3)
    save("EvsZ")
    
    ################################
    plt.figure()
    plt.scatter(data.Z, data.S1i)
    x, y, _ = fitf.profileX(data.Z, data.S1i, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (1., 1e-2, 1e-4))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 20, "{:.3g} + {:.3g} x + {:.3g} x^2".format(*f.values))
    labels("Drift time ($\mu$s)", "S1 charge (pes)")
    plt.ylim(0, 25)
    save("S1vsZ")
    
    ################################
    plt.figure()
    plt.scatter(data.S1i, data.S2i)
    x, y, _ = fitf.profileX(data.S1i, data.S2i, 100, (0, 20))
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (6e3, -1.))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(15, 4200, "{:.3f} + {:.3f} x".format(*f.values))
    labels("S1 charge (pes)", "S2 energy (pes)")
    plt.xlim(0, 20)
    plt.ylim(4e3, 8e3)
    save("S2vsS1")
    
    ################################
    plt.figure()
    plt.scatter(data.Z, data.S1w)
    x, y, _ = fitf.profileX(data.Z, data.S1w, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (1., 1.))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(20, 20, "{:.3f} + {:.3f} x".format(*f.values))
    labels("Drift time ($\mu$s)", "S1 width (ns)")
    plt.ylim(0, 500)
    save("S1widthvsZ")
    
    ################################
    plt.figure()
    plt.scatter(data.Z, data.S1h)
    x, y, _ = fitf.profileX(data.Z, data.S1h, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (1., 0.8, 0.01))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 6, "{:.3g} + {:.3g} x + {:.3g} x^2".format(*f.values))
    labels("Drift time ($\mu$s)", "S1 height (pes)")
    plt.ylim(0, 7)
    save("S1heightvsZ")
    
    ################################
    plt.figure()
    plt.scatter(data.Z, data.S2w)
    x, y, _ = fitf.profileX(data.Z, data.S2w, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.power, x, y, (1., 0.8))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 20, "{:.3f} $\cdot$ x^{:.2f}".format(*f.values))
    labels("Drift time ($\mu$s)", "S2 width ($\mu$s)")
    plt.ylim(0, 30)
    save("S2widthvsZ")
    
    ################################
    plt.figure()
    plt.scatter(data.Z, data.S2h)
    x, y, _ = fitf.profileX(data.Z, data.S2h, 100)
    plt.plot(x, y, profOpt)
    fun = lambda x, *args: fitf.expo(x,*args[:2])/fitf.power(x, *args[2:])
    f = fitf.fit(fun, x, y, (1., -2e4, 0.1, -0.8))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(3e2, 4e3, "{:.3f} exp(x/{:.3g}) / "
                       "({:.3g} $\cdot$ x^{:.2f})".format(*f.values))
    labels("Drift time ($\mu$s)", "S2 height (pes)")
    plt.ylim(0, 5e3)
    save("S2heightvsZ")
    
    ################################
    plt.figure()
    plt.scatter(data.S2w, data.S2h)
    x, y, _ = fitf.profileX(data.S2w, data.S2h, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.power, x, y, (1., -1.0))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(15, 4e3, "{:.3f} $\cdot$ x^{:.2f}".format(*f.values))
    labels("S2 width ($\mu$s)", "S2 height (pes)")
    plt.ylim(0, 5e3)
    save("S2heightvsS2width")


@timefunc
def plot_tracking_info(data, outputfolder="plots/"):
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    save = functools.partial(save_to_folder, outputfolder)

    ################################
    plt.figure()
    plt.hist(data.Nsipm, 10, range=(0, 10))
    labels("Number of SiPMs touched", "Entries")
    save("Nsipm")

    ################################
    plt.figure()
    plt.scatter(data.Z, data.Nsipm)
    x, y, _ = fitf.profileX(data.Z, data.Nsipm, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (2, -1e-5, -1e-2, -1e-3))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(200, 3.5, "{:.3g} + {:.3g} x + {:.3g} x^2 + {:.3g} x^3".format(*f.values))
    labels("Drift time ($\mu$s)", "Number of SiPMs touched")
    save("NsipmvsZ")

    data.apply_mask(data.Nsipm > 0)
    ################################
    plt.figure()
    plt.scatter(data.R, data.Nsipm)
    x, y, _ = fitf.profileX(data.R, data.Nsipm, 50)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.expo, x, y, (1., -0.8))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 3.2, "{:.3f} exp(x/{:.2f})".format(*f.values))
    labels("r (mm)", "Number of SiPMs touched")
    save("NsipmvsR")

    ################################
    plt.figure()
    plt.hist(data.Q, 100, range=(0, 1e3))
    labels("Traking Plane integral (pes)", "Entries")
    save("Q")

    ################################
    plt.figure()
    pdf(data.Q, 100, range=(0, 1e3))
    labels("Traking Plane integral (pes)", "Entries")
    save("Q_log")

    ################################
    plt.figure()
    plt.scatter(data.Z, data.Q)
    x, y, _ = fitf.profileX(data.Z, data.Q, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (400, -1, 0.1, 0.01))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(200, 500, "{:.3g} + {:.3g} x + {:.3g} x^2 + {:.3g} x^3".format(*f.values))
    labels("Drift time ($\mu$s)", "Traking Plane integral (pes)")
    save("QvsZ")

    ################################
    plt.figure()
    ave = f.fn(data.Z[data.Z<200])
    plt.hist((data.Q[data.Z<200] - ave)/ave**0.5, 100, range=(-10, 10))
    labels("Traking Plane integral - average value (pes)", "Entries")
    save("QvsZ_shifted")

    ################################
    plt.figure()
    x, y, q, qe = fitf.profileXY(data.X, data.Y, data.Q, 100, 100)
    x = np.repeat(x, x.size)
    y = np.tile(y, y.size)
    q = np.concatenate(q)
    plt.scatter(x, y, c=q, marker="s")
    plt.colorbar().set_label("Tracking plane integral (pes)")
    labels("x (mm)", "y (mm)")
    save("QvsXY")

    ################################
    plt.figure()
    plt.scatter(data.R, data.Q)
    x, y, _ = fitf.profileX(data.R, data.Q, 50)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (200, -0.001))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 600, "{:.3g} + {:.3g} x".format(*f.values))
    labels("r (mm)", "Traking Plane integral (pes)")
    save("QvsR")

    ################################
    plt.figure()
    plt.scatter(data.Phi, data.Q)
    x, y, _ = fitf.profileX(data.Phi, data.Q, 100)
    plt.plot(x, y, profOpt)
    labels("$\phi$ (rad)", "Traking Plane integral (pes)")
    save("QvsPhi")

    ################################
    plt.figure()
    x, y, q, qe = fitf.profileXY(data.X, data.Y, data.S2i, 100, 100)
    x = np.repeat(x, x.size)
    y = np.tile(y, y.size)
    q = np.concatenate(q)
    plt.scatter(x, y, c=q, marker="s")
    plt.colorbar().set_label("Event energy (pes)")
    labels("x (mm)", "y (mm)")
    save("EvsXY")

    ################################
    plt.figure()
    plt.scatter(data.R, data.S2i)
    x, y, _ = fitf.profileX(data.R, data.S2i, 100)
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (6e3, -1e-2, 1e-2, -1e-3, -1e-4, 1e-6, 1e-7))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(0, 3e3, "{:.3g} + {:.3g} x + {:.3g} x^2 + {:.3g} x^3 + {:.3g} x^4 + {:.3g} x^5 + {:.3g} x^6".format(*f.values))
    labels("r (mm)", "Event energy (pes)")
    save("EvsR")

    ################################
    plt.figure()
    plt.scatter(data.Phi, data.S2i)
    x, y, _ = fitf.profileX(data.Phi, data.S2i, 100)
    plt.plot(x, y, profOpt)
    labels("$\phi$ (rad)", "Event energy (pes)")
    save("EvsPhi")

    ################################
    plt.figure()
    plt.scatter(data.Nsipm, data.Q)
    x, y, _ = fitf.profileX(data.Nsipm, data.Q, 100, xrange=(0, 5))
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.power, x, y, (100, 0.7))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(2.5, 50, "{:.3f} $\cdot$ x^{:.2f}".format(*f.values))
    labels("Number of SiPMs touched", "Traking Plane integral (pes)")
    save("QvsNsipm")

    ################################
    plt.figure()
    plt.hist(data.X, 100, range=(-220, 220))
    labels("x (mm)", "Entries")
    save("X")

    ################################
    plt.figure()
    plt.hist(data.Y, 100, range=(-220, 220))
    labels("y (mm)", "Entries")
    save("Y")

    ################################
    plt.figure()
    plt.hist(data.R, 100, range=(0, 220))
    labels("r (mm)", "Entries")
    save("R")

    ################################
    plt.figure()
    plt.hist(data.Phi, 100, range=(-pi, pi))
    labels("$\phi$ (rad)", "Entries")
    save("Phi")

    ################################
    plt.figure()
    plt.hist2d(data.X, data.Y, 100, range=((-220, 220),
                                           (-220, 220)))
    plt.colorbar().set_label("# events")
    labels("x (mm)", "y (mm)")
    save("XY")

    ################################
    plt.figure()
    plt.hist2d(data.R, data.Phi, 100, range=((  0, 220),
                                             (-pi, pi )))
    plt.colorbar().set_label("# events")
    labels("r (mm)", "$\phi$ (mm)")
    save("RPhi")

    ################################
    plt.figure()
    plt.hist(data.Xrms, 100, range=(0, 50))
    labels("rms x (mm)", "Entries")
    save("rmsX_full")

    ################################
    plt.figure()
    plt.hist(data.Yrms, 100, range=(0, 50))
    labels("rms y (mm)", "Entries")
    save("rmsY_full")

    data.apply_mask((data.Xrms > 1e-3) & (data.Yrms > 1e-3))
    ################################
    plt.figure()
    plt.hist(data.Xrms, 100, range=(0, 50))
    labels("rms x (mm)", "Entries")
    save("rmsX")

    ################################
    plt.figure()
    plt.hist(data.Yrms, 100, range=(0, 50))
    labels("rms y (mm)", "Entries")
    save("rmsY")

    ################################
    plt.figure()
    plt.scatter(data.Z, data.Xrms)
    x, y, _ = fitf.profileX(data.Z, data.Xrms, 100, yrange=(0,30))
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (1., 0.8))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(100, 5, "{:.3f} + {:.3f} x".format(*f.values))
    labels("Drift time ($\mu$s)", "rms x (mm)")
    save("rmsXvsZ")

    ################################
    plt.figure()
    plt.scatter(data.Z, data.Yrms)
    x, y, _ = fitf.profileX(data.Z, data.Yrms, 100, yrange=(0,30))
    plt.plot(x, y, profOpt)
    f = fitf.fit(fitf.polynom, x, y, (1., 0.8))
    plt.plot(x, f.fn(x), fitOpt)
    plt.text(100, 5, "{:.3f} + {:.3f} x".format(*f.values))
    labels("Drift time ($\mu$s)", "rms y (mm)")
    save("rmsYvsZ")



os.environ["IC_DATA"] = os.environ["IC_DATA"] + "/Kr2016/"
pattern = "$IC_DATA/pmaps_NEXT_v0_08_09_Kr_ACTIVE_*_0_7bar__10000.root.h5"
ifiles  = glob.glob(os.path.expandvars(pattern))
#ifiles = ifiles[:1]

data = fill_events(ifiles)
full = Dataset(data)
good = Dataset(data, True)

print("Full set   :", full.evts.size)
print("Reduced set:", good.evts.size)
print("Ratio      :", good.evts.size/full.evts.size)



#plot_S12_info(full)
#plot_evt_info(good)
plot_tracking_info(good)
