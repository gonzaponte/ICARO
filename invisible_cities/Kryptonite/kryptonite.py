import numpy as np
from invisible_cities.database import load_db
import tables as tb
from invisible_cities.reco.pmaps_functions import load_pmaps
from invisible_cities.reco.pmap_io import event_writer, run_writer,\
     _make_run_event_tables

from invisible_cities.core.system_of_units_c import units
from invisible_cities.reco.params import S12Params, ThresholdParams

from pmp_functions import s12_features, charge_and_position_from_S2Si,\
     drift_time, kr_coordinates

from kr_io import KrEvent, kr_writer
from enum import Enum
from collections import namedtuple
from collections import defaultdict

kVerbose = Enum('kVerbose','chat talk quiet mute')

KrSelection = namedtuple('KrSelection',
              's2_multiplicity')


class Krypton:
    """
    A krypton pre-proc analysis.
    Reads PMAPS and select events to be written to DF-DST
    """
    def __init__(self,
                 run_number,
                 input_files,
                 output_file,
                 s1par,
                 s2par,
                 thresholds,
                 krypton_selection,
                 ):

        self.input_files = input_files
        self.output_file = output_file
        self.s1par = s1par
        self.s2par = s2par
        self.thr = thresholds
        self.krsel = krypton_selection
        self.events_selected = {}
        self.events_selected = defaultdict(lambda:0,self.events_selected)

        DataSiPM = load_db.DataSiPM(run_number)
        self.xs = DataSiPM.X.values
        self.ys = DataSiPM.Y.values
        self.s1f = {}
        self.s2f = {}


    def increment_counter(self, counter_name):
        counter = self.events_selected[counter_name] + 1
        self.events_selected[counter_name] = counter


    def select_peaks_(self, s12, s12f, threshold, lmin, lmax):
        """Select peaks.

        Only consider peaks with energy above thr_s12 threshold and
        width between lmin and lmax.

        """
        PEAKS = []
        for peak, (t, E) in s12.items():
            s12f[peak] = s12_features(s12, peak_number=peak)
            E2 = E[np.where(E > threshold)]
            if len(E2) < lmin or len(E2) > lmax:
                continue
            PEAKS.append(peak)

        return PEAKS

    def select_s1(self, s1):
        """Select s1 signals."""
        PEAKS = self.select_peaks_(s1, self.s1f, self.thr.thr_s1,
                                   self.s1par.lmin, self.s1par.lmax)

        if len(PEAKS) == 1:
            return True, PEAKS
        else:
            return False, PEAKS

    def select_s2(self, s2):
        """Select s2 signals."""
        PEAKS = self.select_peaks_(s2, self.s2f, self.thr.thr_s2,
                                   self.s2par.lmin/40,   # in mus
                                   self.s2par.lmax/40)
        if len(PEAKS) >= 1 and len(PEAKS) <= self.krsel.s2_multiplicity:
            return True, PEAKS
        else:
            return False, PEAKS


    def run(self, drift_velocity= 0.9 * units.mm/units.mus,
                  printout = 10,
                  nmax = 10,
                  verbose  = kVerbose.mute):
        """Loop over events, select and write table."""

        kre = KrEvent(drift_velocity= drift_velocity)
        n_events_tot = 0

        with kr_writer(self.output_file) as write:

            for ffile in self.input_files:
                print("Opening", ffile, end="... ")
                filename = ffile

                events = 0
                timestamp = 0
                with tb.open_file(filename, "r") as h5in:
                    events = h5in.root.Run.events.col('evt_number')
                    timestamp = h5in.root.Run.events.col('timestamp')

                if verbose == kVerbose.chat:
                    print('events for file = {}'.format(
                           events))

                S1, S2, S2Si = load_pmaps(filename)
                # event loop
                for i, evt_number in enumerate(events):
                    n_events_tot +=1
                    if n_events_tot == nmax:
                        return nmax
                    self.increment_counter('n_tot')
                    if n_events_tot%printout ==0 :
                        print('reading event {} with event number = {} '.format(
                               n_events_tot, evt_number))

                    if evt_number not in S1:
                        self.increment_counter('not_s1')
                        if verbose == kVerbose.talk or verbose == kVerbose.chat:
                            print('S1 not found for event = {}'.format(
                                   evt_number))
                        continue

                    cond, S1_peaks  = self.select_s1(S1[evt_number])
                    if not cond:
                        self.increment_counter('s1_not_1')
                        if verbose == kVerbose.talk or verbose == kVerbose.chat:
                            print('S1 not one for event number = {}'.format(
                                   evt_number))
                        continue
                    self.increment_counter('s1_eq_1')

                    if evt_number not in S2:
                        self.increment_counter('not_s2')
                        if verbose == kVerbose.talk or verbose == kVerbose.chat:
                            print('S2 not found for event = {}'.format(
                                   evt_number))
                        continue

                    cond, S2_peaks = self.select_s2(S2[evt_number])
                    if not cond:
                        self.increment_counter('s2_not_eq_s2_mult')
                        if verbose == kVerbose.talk or verbose == kVerbose.chat:
                            print('S2 larger than {} for event = {}'.format(
                               self.krsel.s2_multiplicity, evt_number))
                        continue
                    self.increment_counter('s2_eq_s2_mult')

                    if evt_number not in S2Si:
                        self.increment_counter('not_si')
                        if verbose == kVerbose.talk or verbose == kVerbose.chat:
                            print('S2Si not found for event = {}'.format(
                                                       evt_number))
                        continue


                    for peak in S2_peaks:
                        if verbose == kVerbose.chat:
                            print('S2 peak = {}'.format(peak))

                        s1 = S1[evt_number]
                        s2 = S2[evt_number]
                        s2si = S2Si[evt_number]

                        if verbose == kVerbose.chat:
                            print('S1 = {}'.format(s1))
                            print('S2 = {}'.format(s2))
                            print('Si = {}'.format(s2si))


                        xsipm, ysipm, Q = charge_and_position_from_S2Si(s2si,
                                                          self.xs,
                                                          self.ys, peak_number=peak)

                        dt              =     drift_time(self.s1f[S1_peaks[0]],
                                                          self.s2f,
                                                          peak_number=peak)
                        # fill krypton event
                        krc = kr_coordinates(xsipm,
                                             ysipm,
                                             Q,
                                             dt,
                                             drift_velocity)
                        if verbose == kVerbose.talk\
                        or verbose == kVerbose.chat:
                            print('kr coordinates = {}'.format(krc))

                        kre.add_s1_features(s1,   peak_number=S1_peaks[0])
                        kre.add_s2_features(s2,   peak_number=peak)
                        kre.add_nsipm      (s2si, peak_number=peak)
                        kre.add_coordinates(krc)
                        kre.add_ns1_pmt    (0)

                        # write to file
                        write(run_number, evt_number, peak, timestamp[i], kre)

        return n_events_tot

if __name__ == "__main__":
    import sys
    import os
    import time
    import glob
    from invisible_cities.reco.params import S12Params, ThresholdParams
    from   invisible_cities.core.system_of_units_c import units


    run_number = 3389
    input_dir = os.path.join(os.environ['IC_DATA'],
                'Kr2016/data/{}'.format(run_number))
    files_in = glob.glob(os.path.join(input_dir,'*.h5'))
    files_in.sort()
    print(files_in)
    #input_file = os.path.join(os.environ['IC_DATA'],
    #    'LSC/pmaps/{}/pmaps_waves.gdcsnext.000_{}.root.h5'.format(run_number,
    #                                                              run_number))
    #files_in = glob.glob(input_file )
    #files_in.sort()

    file_out = os.path.join(os.environ['IC_DATA'],
        'kdst_{}_full.h5'.format(run_number))

    s1par  = S12Params(tmin=0*units.mus, tmax=640*units.mus,
                       lmin=5, lmax=20,
                       stride=4, rebin=False)
    s2par = S12Params(tmin=640*units.mus, tmax=670*units.mus,
                      lmin=80, lmax=20000,
                      stride=40, rebin=True)

    thr   = ThresholdParams(thr_s1=0.5 * units.pes,  thr_s2=1 *units.pes,
                        thr_MAU = 3 * units.adc, thr_sipm = 3.5 * units.pes,
                        thr_SIPM = 30 * units.adc)

    krs = KrSelection(s2_multiplicity=1)

    krp = Krypton(run_number,
                  files_in,
                  file_out,
                  s1par,
                  s2par,
                  thr,
                  krs)
    krp.run(drift_velocity= 0.9 * units.mm/units.mus, printout=100,
            nmax=10000000,
            verbose=kVerbose.quiet)

    print(krp.events_selected)
