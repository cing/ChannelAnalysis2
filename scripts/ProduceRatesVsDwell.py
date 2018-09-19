""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted. """

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from Initialize import Project
import itertools
from collections import defaultdict
from Transitions import mean_unique_state_transition_counts
from Plot import plot_logseries_with_error
import matplotlib.pyplot as plt

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def prune_time(df, start, end):
    return df[(df["Time"] >= start) & (df["Time"] <= end)]

def prune_column(df, col, start, end):
    return df[(df[col] >= start) & (df[col] <= end)]

if __name__ == '__main__':
    parser = ArgumentParser(
    description='This script extracts basic transition statistics of \
    coordination with attention to statistics across multiple timeseries.')
    parser.add_argument(
    '-c', dest='cfg_path', nargs="+", type=str, required=True,
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', nargs="+", type=str, required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    f1, ax1 = plt.subplots()

    # TODO, just make it an argument dude
    min_dwell = 1
    max_dwell = 51
    for cfg, series_name in zip(args.cfg_path,args.series_name):
        # This builds a Project using the configuration file argument
        TestProject = Project(cfg)
        print("Successfully Loaded Project: %s" % TestProject.name)

        # default = '1st Shell Na+ SF Coordination'
        coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                      TestProject.start_time,
                                      TestProject.end_time)

        if True:
            mean_rates_per_dwell_time = []
            sem_rates_per_dwell_time = []
            for dwell_cut in range(min_dwell, max_dwell):
                rate_stats_per_dwell_time = mean_unique_state_transition_counts(coord_df_noequil,
                                                      TestProject.state_dividers,
                                                      state_names=TestProject.state_names,
                                                      dwell_cut=dwell_cut,
                                                      compute_rates=True)

                mean_rates_per_dwell_time.append(rate_stats_per_dwell_time["Mean"])
                sem_rates_per_dwell_time.append(rate_stats_per_dwell_time["SEM"])

        joint_mean_rates_per_dwell = pd.concat(mean_rates_per_dwell_time, axis=1)
        joint_mean_rates_per_dwell.columns = [str(x)+" Dwell" for x in range(min_dwell,max_dwell)]

        joint_sem_rates_per_dwell = pd.concat(sem_rates_per_dwell_time, axis=1)
        joint_sem_rates_per_dwell.columns = [str(x)+" Dwell" for x in range(min_dwell,max_dwell)]


        trans_vs_lag = pd.concat([1000*joint_mean_rates_per_dwell.ix['E-EL'],
                                  1000*joint_sem_rates_per_dwell.ix['E-EL']], axis=1)
        trans_vs_lag.index = 1+np.arange(len(joint_mean_rates_per_dwell.ix['E-EL']))
        trans_vs_lag.columns = ["Data","Error"]

        print(trans_vs_lag)
        plot_logseries_with_error(ax1, trans_vs_lag, xlim=[0,max_dwell-1],
                                   color=TestProject.color_ts[series_name],
                                   ylim=[1,2000],
                                   x_label="minimum dwell time (steps)",
                                   y_label="transitions per microsecond")

    f1.set_size_inches(18.5, 18.5)
    f1.savefig(TestProject.output_name+'_E_to_EL_trans_vs_dwell.pdf', dpi=200)
