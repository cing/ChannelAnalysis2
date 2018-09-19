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
from Transitions import unique_state_transition_counts
from Plot import plot_histogram_from_ts
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
    '-d', dest='dwell_cut', type=int, required=True,
    help='minimum number of timesteps in state to count as transition')
    parser.add_argument(
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    f1, ax1 = plt.subplots()

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
        print(ioncount_norm)

        # Plot the ion histogram
        plot_histogram_from_ts(ax1, coord_df_noequil["Z"], bins=240, xlim=[-10,14],
                                   ylim=[0,0.8], color=TestProject.color_ts[series_name],
                                   hist_range=[-10,14], norm_factor=norm_factor,
                                   hist_vline=np.array(TestProject.state_dividers),
                                   x_label="position (Ang)", y_label="probability (arb. units)")

        # Plot the rates extracted from transcount
        transcount = unique_state_transition_counts(coord_df_noequil,
                                            TestProject.state_dividers,
                                            state_names=TestProject.state_names,
                                            dwell_cut=args.dwell_cut,
                                            compute_rates=True)

        trans_dict=(1000*transcount).to_dict()
        print(trans_dict)
        pairs=["-".join(pair) for pair in list(window(TestProject.state_names))]
        pairs_reverse=["-".join(reversed(pair)) for pair in list(window(TestProject.state_names))]

        for div1, div2, pos in zip(pairs, pairs_reverse, np.array(TestProject.state_dividers)-1):
            try:
                avg_trans = np.mean([trans_dict[div1], trans_dict[div2]])
                ax1.text(pos, 0.70-0.2*series_num, str(np.around(avg_trans,1))+ "/$\mu$s",
                color=TestProject.color_ts[series_name], size=16,
                bbox=dict(facecolor='white',
                edgecolor=TestProject.color_ts[series_name], pad=10.0))
            except KeyError:
                ax1.text(pos, 0.70-0.2*series_num, str(0.0)+ "/$\mu$s",
                color=TestProject.color_ts[series_name], size=16,
                bbox=dict(facecolor='white',
                edgecolor=TestProject.color_ts[series_name], pad=10.0))

    f1.set_size_inches(18.5, 5.5)
    f1.savefig(TestProject.output_name+'_zhistogram_w_dividers.pdf', dpi=200)
