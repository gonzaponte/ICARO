"""
Kr analysis
"""
import os
import functools
import time
import glob
print(time.asctime())

import tables as tb
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = 12, 10

import invisible_cities.database.load_db as DB
import invisible_cities.core.system_of_units_c as SystemOfUnits
import invisible_cities.core.pmaps_functions as pmapf
import invisible_cities.core.fit_functions as fitf

DataPMT = DB.DataPMT()
DataSiPM = DB.DataSiPM()
units = SystemOfUnits.SystemOfUnits()

def width(times):
    return (np.max(times) - np.min(times)) / units.ns


def plot_pmap(s1, s2, evt):
    plt.plot(s1[s1.event==evt].time, s1[s1.event==evt].ene)
    plt.plot(s2[s2.event==evt].time, s2[s2.event==evt].ene)


def flat(list2d):
    if hasattr(list2d[0],"__iter__"):
        return list(it.chain.from_iterable(list2d))
    else:
        return list(list2d)


def fine(x):
    """
    Produces a fine-grained array for plotting.
    """
    return np.linspace(np.min(x), np.max(x), 1000)

inputfile = os.path.expandvars("$GITDIR/NEXTdata/Kr2016/KrMC.h5")

h5f = tb.open_file(inputfile)
print("H5File", h5f)
print("S1 colnames", h5f.root.PMAPS.S1.colnames)
print("S2 colnames", h5f.root.PMAPS.S2.colnames)
print("Si colnames", h5f.root.PMAPS.S2Si.colnames)
h5f.close()


profOpt = "--k"
fitOpt  = "r"


class Event:
    def __init__(self):
        self.nS1 = 0
        self.S1w = []
        self.S1h = []
        self.S1i = []
    
        self.nS2 = 0
        self.S2w = []
        self.S2h = []
        self.S2i = []
    
        self.X   = 1e3
        self.Y   = 1e3
        self.Z   = -1
        self.R   = -1
        
        self.ok  = False


class Dataset:
    def __init__(self, evts, mask=False):
        print("Creating dataset with mask", mask)
        t0 = time.time()
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
        print(time.time() - t0)


def fill_events(inputfiles):
    evts_out = []
    for ifile in inputfiles:
        s1s, s2s, sis = pmapf.read_pmaps(ifile)
        evts = set(s1s.evtDaq)|set(s2s.evtDaq)
        nevt = len(evts)
        print(ifile, nevt)
        t0 = time.time()
        for i, evt_ in enumerate(evts):
            evt = Event()
            s1  = s1s[s1s.evtDaq == evt_]
            s2  = s2s[s2s.evtDaq == evt_]

            s1peaks = set(s1.peak.values)
            s2peaks = set(s2.peak.values)
            
            nS1 = len(s1peaks)
            nS2 = len(s2peaks)
            
            evt.nS1 = nS1
            evt.nS2 = nS2
            
            s1time = 0
            for peak_ in s1peaks:
                peak  = s1[s1.peak == peak_]
                times = peak.time.values
                amps  = peak.ene.values
                evt.S1w.append(width(times))
                evt.S1h.append(np.max(amps))
                evt.S1i.append(np.sum(amps))
                s1time = times[np.argmax(amps)]
            
            s2time = 0
            for peak in s2peaks:
                peak  = s2[s2.peak == peak]
                times = peak.time.values
                amps  = peak.ene.values

                evt.S2w.append(width(times) * units.ns/units.mus)
                evt.S2h.append(np.max(amps))
                evt.S2i.append(np.sum(amps))
                s2time = times[np.argmax(amps)]

            evt.ok = nS1 == nS2 == 1
            if evt.ok:
                evt.Z = (s2time - s1time) * units.ns / units.mus
            evts_out.append(evt)
        print(time.time() - t0, "s")
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


def plot_S12_info(data, outputfolder="plots/"):
    print("Entering plot_S12_info")
    if not os.path.exists(outputfolder):
        os.mkdir(outputfolder)
    save = functools.partial(save_to_folder, outputfolder)
    plt.figure()
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


def plot_evt_info(data, outputfolder="plots/"):
    print("Entering plot_evt_info")
    save = functools.partial(save_to_folder, outputfolder)
    plt.figure()
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
    f = fitf.fit(lambda x, *args: fitf.expo(x,*args[:2])/fitf.power(x, *args[2:]), x, y, (1., -2e4, 0.1, -0.8))
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


#os.environ["IC_DATA"] = os.environ["IC_DATA"] + "/Kr2016/"
pattern = "$IC_DATA/pmaps_NEXT_v0_08_09_Kr_ACTIVE_*_0_7bar__10000.root.h5"
ifiles = glob.glob(os.path.expandvars(pattern))
#ifiles = ifiles[:1]
data = fill_events(ifiles)
full = Dataset(data)
good = Dataset(data, True)

print("Full set:", full.evts.size)
print("Reduced set:", good.evts.size)
print("Ratio:", good.evts.size/full.evts.size)


t0 = time.time()
plot_S12_info(full)
print(time.time()-t0)

t0 = time.time()
plot_evt_info(good)
print(time.time()-t0)