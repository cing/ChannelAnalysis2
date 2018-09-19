""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted. """

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from PandasMD.Initialize import Project
import itertools
from collections import defaultdict
from PandasMD.Plot import plot_ts_mean
from PandasMD.Coordination import macrostate_labels
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

    f1, ax = plt.subplots(1)
    target_ax = ax

    if len(args.cfg_path) == 1:
        # This builds a Project using the configuration file argument
        TestProject = Project(args.cfg_path, coord_ts=args.series_name)
        #TestProject = Project(args.cfg_path)
        print("Successfully Loaded Project: %s" % TestProject.name)

        for series_name in args.series_name:
            coord_df2_trim = prune_column(TestProject.coord_ts[series_name],
                                          "Z", -10, 14)
            lbl = macrostate_labels(coord_df2_trim)
            ts = lbl[lbl["Order"]==0].pivot(index='Frame', columns= 'TrajNum', values="Occupancy")
            ts_mean = ts.mean(axis=1)
            ts_sem = ts.sem(axis=1)
            plot_ts_mean(ax, ts_mean, ts_sem, xlim=[0,1000], ts_vline=150, skip=2,
                         dt=TestProject.dt, color=TestProject.color_ts[series_name])

    else:
        for cfg, series_name in zip(args.cfg_path,args.series_name):
            # This builds a Project using the configuration file argument
            # TestProject = Project(cfg)
            TestProject = Project(cfg, coord_ts=[series_name,])
            print("Successfully Loaded Project: %s" % TestProject.name)

            coord_df2_trim = prune_column(TestProject.coord_ts[series_name],
                                          "Z", -10, 14)
            lbl = macrostate_labels(coord_df2_trim)
            ts = lbl[lbl["Order"]==0].pivot(index='Frame', columns= 'TrajNum', values="Occupancy")
            ts_mean = ts.mean(axis=1)
            ts_sem = ts.sem(axis=1)
            plot_ts_mean(ax, ts_mean, ts_sem, xlim=[0,1000], ts_vline=150, skip=2,
                         dt=TestProject.dt, color=TestProject.color_ts[series_name])

    f1.set_size_inches(12.5, 5.5)
    f1.savefig(TestProject.output_name+'_channelocc_vs_time.pdf', dpi=200)
