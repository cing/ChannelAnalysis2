""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted. """

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from ChannelAnalysis2.Initialize import Project
import itertools
from collections import defaultdict
from ChannelAnalysis2.Plot import plot_order_histograms, plot_histogram_from_ts
from ChannelAnalysis2.Coordination import species_macrostate_labels
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
    '-ts', dest='series_name', type=str, required=True, nargs="+",
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path, coord_ts=args.series_name)
    print("Successfully Loaded Project: %s" % TestProject.name)

    all_series = []
    for series_id, series in enumerate(args.series_name):

        coord_df_noequil = prune_time(TestProject.coord_ts[series],
                                      TestProject.start_time,
                                      TestProject.end_time)
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 10) 
        all_series.append(coord_df2_trim)

    superdf = pd.concat(all_series)

    frames = 6

    ioncount_norm = superdf.groupby(["TrajNum","Time"]).size().mean() 
    histogram, edges = np.histogram(superdf["Z"], bins=300, range=[-10,14], normed=False)
    ioncount_norm = 1
    norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
    #print(norm_factor)

    coord_df2_order = species_macrostate_labels(superdf)

    f1, ax = plt.subplots(frames)
    plot_order_histograms(ax, coord_df2_order, prime=False, xlim=[-10,14], ylim=[0,1.0],
                            x_label="position (Ang)",
                            y_label="probability (arb. units)",
                            norm_factor=norm_factor)

    f1.set_size_inches(12.5, 5.5*frames)
    f1.savefig(TestProject.output_name+'_zhistogram_for_ordered_occ_noprime_occnorm_mixed.pdf', dpi=200)

    f2, ax = plt.subplots(frames)
    plot_order_histograms(ax, coord_df2_order, prime=True, xlim=[-10,14], ylim=[0,1.0],
                            x_label="position (Ang)",
                            y_label="probability (arb. units)",
                            norm_factor=norm_factor)

    f2.set_size_inches(12.5, 5.5*frames)
    f2.savefig(TestProject.output_name+'_zhistogram_for_ordered_occ_prime_occnorm_mixed.pdf', dpi=200)
