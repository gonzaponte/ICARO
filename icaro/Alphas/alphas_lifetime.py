# A script to compute alpha lifetime.
import os
import sys
import argparse
import datetime

import numpy             as np
import matplotlib.pyplot as plt

import invisible_cities.core.core_functions as coref
import invisible_cities.reco. dst_functions as dstf
import invisible_cities.core.fit_functions  as fitf

from icaro.core.kdst_functions import event_rate
from icaro.core.kdst_functions import profile_and_fit
from icaro.core.kdst_functions import time_from_timestamp
from icaro.core.kdst_functions import to_deltatime
from icaro.core.kdst_functions import lifetime_vs_t
from icaro.core.kdst_functions import save_lifetime

plt.rcParams["figure.figsize"         ] = 10, 8
plt.rcParams["font.size"              ] = 14
plt.rcParams["figure.max_open_warning"] = 1000

space = "\n.\n.\n.\n"
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), end=space)


## Configuration
program, *args = sys.argv
parser = argparse.ArgumentParser(program)
parser.add_argument("-r", metavar="run_numbers", type=int, help="run numbers"  , nargs='+')
parser.add_argument("-i", metavar="input path" , type=str, help="input path"   )
parser.add_argument("-o", metavar="database"   , type=str, help="database file", default="$ICARODIR/Alphas/Litetimes.txt")
parser.add_argument("-c", metavar="comment"    , type=str, help="comments"     , default="")
parser.add_argument("-p", metavar="plotsfolder", type=str, help="plots folder" )
parser.add_argument("--save-plots", action="store_true"  , help="store control plots" )
parser.add_argument("--overwrite" , action="store_true"  , help="overwrite datebase values" )

flags, extras = parser.parse_known_args(args)
flags.i = os.path.expandvars(flags.i if flags.i else "$ALPHASDIR")
flags.o = os.path.expandvars(flags.o if flags.o else "$ICARODIR/icaro/Alphas/Lifetimes.txt")
flags.p = os.path.expandvars(flags.p if flags.p else "./")

run_numbers   = flags.r
data_filename = flags.i + "{0}/dst_{0}.root.h5"
text_filename = flags.o
run_comment   = flags.c
plots_folder  = flags.p
save_plots    = flags.save_plots
overwrite     = flags.overwrite

def savefig(name, run_number):
    folder = os.path.join(plots_folder, str(run_number))
    if not save_plots: return
    if not os.path.exists(plots_folder): os.mkdir(plots_folder)
    if not os.path.exists(      folder): os.mkdir(      folder)
    plt.savefig(os.path.join(folder, name + ".png"))


for run_number in run_numbers:
    full       = dstf.load_dst(data_filename.format(run_number), "DST", "Events")
    t_begin    = np.min(full.time)
    t_end      = np.max(full.time)
    run_dt     = t_end - t_begin
    full.time -= t_begin
    fid        = full[full.R < 100] # michel sorel cuts
    cath       = full[full.Z > 500]
    bulk       = full[full.Z < 500]

    print("Run number            :", run_number)
    print("# alphas              :", len(full) )
    print("# alphas at R < 100 mm:", len(fid ) )
    print("# alphas at Z > 500 µs:", len(cath) )
    print("# alphas at Z < 500 µs:", len(bulk) )


    #--------------------------------------------------------
    plt.figure()
    n, bins, _ = \
    plt.hist  (full.time/60, 40)
    plt.xlabel("Time (min)")
    plt.ylabel("Rate (({:.1f} min)$^{{-1}}$)".format(np.diff(bins)[0]))
    plt.title ("Trigger rate")
    savefig   ("TriggerRate", run_number)
    
    rate = event_rate(full)
    print("Average trigger rate: {:.2f} evts/s".format(rate))


    #--------------------------------------------------------
    plt.figure (figsize=(12, 8))
    plt.subplot(2, 3, 1)
    plt.hist   (full.S1e, 50, (0, np.max(full.S1e)))
    plt.xlabel ("S1 energy (pes)")
    plt.ylabel ("Entries")
    plt.title  ('S1 energy')

    plt.subplot(2, 3, 2)
    plt.hist   (full.S2e, 50, (0, np.max(full.S2e) * 1.2))
    plt.xlabel ("S2 energy (pes)")
    plt.ylabel ("Entries")
    plt.title  ('S2 energy')

    plt.subplot(2, 3, 3)
    plt.hist   (fid.S2e, 50, (0, np.max(full.S2e) * 1.2))
    plt.xlabel ("S2 energy (pes)")
    plt.ylabel ("Entries")
    plt.title  ('S2 energy (R < 100 mm)')

    plt.subplot(2, 3, 4)
    plt.hist   (full.Z, 25, (0, 600))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("Entries")
    plt.title  ('Drift time')

    plt.subplot(2, 3, 5)
    plt.hist   (fid.Z, 25, (500, 600))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("Entries")
    plt.title  ('Drift time (cathode)')

    plt.subplot(2, 3, 6)
    plt.hist   (fid.Z, 25, (0, 500))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("Entries")
    plt.title  ('Drift time (bulk)')

    plt.tight_layout()
    savefig("EZ_distributions", run_number)


    #--------------------------------------------------------
    plt.figure()
    plt.hist2d(full.time, full.S2e, (25, 25))
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (pes)")
    plt.title ("Energy vs time")
    savefig   ("EvsT", run_number)


    #--------------------------------------------------------
    plt.figure(figsize=(12, 8))
    zrange  = 0, 610
    S2range = 0, 4e4
    S1range = 0, 3e3

    plt.subplot(2, 3, 1)
    plt.hist2d (full.Z, full.S2e, (25, 25), range=(zrange, S2range))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("S2 energy (pes)")
    plt.title  ('S2 vs Z')

    plt.subplot(2, 3, 2)
    plt.hist2d (full.Z, full.S1e, (25, 25), range=(zrange, S1range))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("S1 energy (pes)")
    plt.title  ('S1 vs Z')

    plt.subplot(2, 3, 3)
    plt.hist2d (full.S2e, full.S1e, (25, 25), range=(S2range, S1range))
    plt.xlabel ("S2 energy (pes)")
    plt.ylabel ("S1 energy (pes)")
    plt.title  ('S1 vs S2')

    plt.subplot(2, 3, 4)
    plt.hist2d (fid.Z, fid.S2e, (25, 25), range=(zrange, S2range))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("S2 energy (pes)")
    plt.title  ('S2 vs Z (R < 100 mm)')

    plt.subplot(2, 3, 5)
    plt.hist2d (fid.Z, fid.S1e, (25, 25), range=(zrange, S1range))
    plt.xlabel ("Drift time ($\mu$s)")
    plt.ylabel ("S1 energy (pes)")
    plt.title  ('S1 vs Z (R < 100 mm)')

    plt.subplot(2, 3, 6)
    plt.hist2d (fid.S2e, fid.S1e, (25, 25), range=(S2range, S1range))
    plt.xlabel ("S2 energy (pes)")
    plt.ylabel ("S1 energy (pes)")
    plt.title  ('S1 vs S2 (R < 100 mm)')

    plt.tight_layout()
    savefig("S12", run_number)


    #--------------------------------------------------------
    xrange  = -215, 215
    yrange  = -215, 215
    xyrange = xrange, yrange
    xybins  = 50, 50

    plt.figure (figsize=(12,10))
    plt.subplot(2, 2, 1)
    plt.hist2d (full.X, full.Y, xybins, xyrange)
    plt.xlabel ("X (mm)")
    plt.ylabel ("Y (mm)")
    plt.title  ('XY distribution')

    plt.subplot(2, 2, 2)
    plt.hist2d (fid.X, fid.Y, xybins, xyrange)
    plt.xlabel ("X (mm)")
    plt.ylabel ("Y (mm)")
    plt.title  ('XY distribution (R < 100 mm)')

    plt.subplot(2, 2, 3)
    plt.hist2d (bulk.X, bulk.Y, xybins, xyrange)
    plt.xlabel ("X (mm)")
    plt.ylabel ("Y (mm)")
    plt.title  ('XY distribution (bulk)')

    plt.subplot(2, 2, 4)
    plt.hist2d (cath.X, cath.Y, xybins, xyrange)
    plt.xlabel ("X (mm)")
    plt.ylabel ("Y (mm)")
    plt.title  ('XY distribution (cathode)')

    plt.tight_layout()
    savefig("XY", run_number)

    
    #--------------------------------------------------------
    zrange =  50, 400
    Erange = 1e4, 7e4
    nbins  =  50
    seed   = Erange[1], zrange[1]/np.log(Erange[0]/Erange[1])

    plt.figure()
    F, x, y, sy = profile_and_fit(fid.Z, fid.S2e, 
                                  xrange = zrange, 
                                  yrange = Erange, 
                                  nbins  = nbins, 
                                  fitpar = seed,
                                  label  = ("Drift time ($\mu$s)", "S2 energy (pes)"))
    savefig("LifetimeFit", run_number)

    print("Energy at Z=0: {:.0f} +- {:.0f} pes".format( F.values[0], F.errors[0]))
    print("Lifetime     : {:.1f} +- {:.1f} µs ".format(-F.values[1], F.errors[1]))
    print("Chi2 fit     : {:.2f}              ".format( F.chi2))
    LT, LTu = -F.values[1], F.errors[1]

    
    #--------------------------------------------------------
    plt.figure()
    dst        = fid[coref.in_range(fid.Z, *zrange)]
    timestamps = list(map(time_from_timestamp, dst.time))

    lifetime_vs_t(dst, nslices=8, timestamps=timestamps)
    plt.xlabel   ("Time (s)")
    plt.ylabel   ("Lifetime (µs)")
    plt.title    ("Lifetime evolution within run");
    savefig      ("LifetimeT", run_number)


    #--------------------------------------------------------    
    def get_centers(xbins):
        # xbins should be shifted, as it contains the lower bounds
        # and we want bin centers
        return xbins[:-1] + np.diff(xbins) * 0.

    plt.figure()

    values, bins = np.histogram(cath.Z,50)
    mean_val = get_centers(bins)[np.argmax(values)]

    zrange   =  mean_val-5, mean_val+5
    nbins    =  50
    y, x, _  = plt.hist(cath.Z,nbins, zrange)
    x        = get_centers(x)

    F        = fitf.fit(fitf.gauss, x, y, (500, mean_val, 2))#, sigma=np.sqrt(y))

    v_drift  = 532./F.values[1]
    ev_drift = v_drift*F.errors[1]/F.values[1]

    print("Drift length   : {:.3f} +- {:.3f} pes".format( F.values[1], F.errors[1]))
    print("Drift velocity : {:.7f} +- {:.1e} µs ".format( v_drift    , ev_drift))

    #--------------------------------------------------------

    plt.plot     (x, F.fn(x), lw=3, c='r')
    plt.xlabel   ('Drift length ($\mu$s)')
    plt.ylabel   ('Entries')
    plt.title    ('Drift length')
    savefig      ("Drift:length", run_number)


    #--------------------------------------------------------
    date_begin = time_from_timestamp(t_begin)
    date_end   = time_from_timestamp(t_end  )
    date_lapse = to_deltatime       (t_begin, t_end, unit="s", to_str=True)

    save_lifetime(text_filename,
                     run_number,       LT,        LTu,
                        t_begin,    t_end,     run_dt,
                     date_begin, date_end, date_lapse,
                        v_drift, ev_drift,
                     comment   = run_comment.replace(" ", "_"),
                     delimiter = " ",
                     overwrite = overwrite)
    print("", end=space)
