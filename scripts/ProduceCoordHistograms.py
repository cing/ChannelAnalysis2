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
from PandasMD.Coordination import coordination_labels
from PandasMD.Transitions import unique_state_transition_counts
from PandasMD.Plot import plot_coord_histogram
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
    '-c', dest='cfg_path', type=str, required=True,
    help='a configfile describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    f1, ax = plt.subplots(len(args.series_name))
    for series_num, series_name in enumerate(args.series_name):
        # default = '1st Shell Na+ SF Coordination'
        coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                      TestProject.start_time,
                                      TestProject.end_time)

        # Extract the mean number of ions in the range -2.0 to 2.0
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
        ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean() 
        histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
        norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))

        coord_colnames = ["S178","E177","L176","T175"]
        coord_df2_trim["CoordLabel"] = coordination_labels(coord_df2_trim, coord_colnames)

        # Probably a better way to do this...
        if len(args.series_name) > 1:
            target_ax = ax[series_num]
        else:
            target_ax = ax

        plot_coord_histogram(target_ax, coord_df2_trim, xlim=[-8,10], ylim=[0,1.0],
                             x_label="position (Ang)",
                             y_label="probability (arb. units)",
                             coord_colnames=coord_colnames, norm_factor=norm_factor)

    f1.set_size_inches(18.5, 5.5*len(args.series_name))
    f1.savefig(TestProject.output_name+'_zhistogram_for_coordlabels.pdf', dpi=200)
