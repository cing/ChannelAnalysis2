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
from PandasMD.Transitions import unique_state_transition_counts
from PandasMD.Plot import plot_histogram_from_ts
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
    '-d', dest='dwell_cut', type=int, nargs="+", required=True,
    help='minimum number of timesteps in state to count as transition')
    parser.add_argument(
    '-ts', dest='series_name', type=str, required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path, coord_ts=[args.series_name])
    print("Successfully Loaded Project: %s" % TestProject.name)

    f1, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylim([0,80])
    #state_dividers = np.arange(-1.0,1.4,0.20)
    #state_names = list(itertools.islice(map(''.join,
    #                                        itertools.product('ABCDEFGHIJKLMNOPQRSTUVWXYZ',
    #                                        repeat=2)),
    #                                    len(state_dividers)+1))

    state_dividers = TestProject.state_dividers
    state_names = TestProject.state_names
    print(state_names)
    print(state_dividers)

    hist_vline=np.array(state_dividers)
    for xval in hist_vline:
        ax1.axvline(xval, color='red', alpha=0.25, linewidth=5)

    # default = '1st Shell Na+ SF Coordination'
    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name],
                                  TestProject.start_time,
                                  TestProject.end_time)

    # Extract the mean number of ions in the range -2.0 to 2.0
    coord_df2_trim = prune_column(coord_df_noequil, "Z", -14, 14)
    ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean() 
    histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
    norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
    print(ioncount_norm)

    # Plot the ion histogram
    plot_histogram_from_ts(ax1, coord_df_noequil["Z"], bins=240, xlim=[-10,14],
                               ylim=[0,0.8], color=TestProject.color_ts[args.series_name],
                               hist_range=[-10,14], norm_factor=norm_factor,
                               x_label="position (Ang)", y_label="probability (arb. units)")

    for dwell_num, dwell_cut in enumerate(args.dwell_cut):

        # Plot the rates extracted from transcount
        transcount = unique_state_transition_counts(coord_df_noequil,
                                            state_dividers,
                                            state_names=state_names,
                                            dwell_cut=dwell_cut,
                                            compute_rates=True)

        trans_dict=(1000*transcount).to_dict()
        pairs=["-".join(pair) for pair in list(window(state_names))]
        pairs_reverse=["-".join(reversed(pair)) for pair in list(window(state_names))]

        x_vals=[]
        y_vals=[]
        for div1, div2, pos in zip(pairs, pairs_reverse, np.array(state_dividers)-0.5):
            x_vals.append(pos+0.5)
            try:
                avg_trans = np.mean([trans_dict[div1], trans_dict[div2]])
                y_vals.append(avg_trans)
                ax1.text(pos, 0.80-0.05*dwell_num, str(np.around(avg_trans,1))+ "/$\mu$s",
                color=TestProject.color_ts[args.series_name], size=12,
                bbox=dict(facecolor='white',
                edgecolor=TestProject.color_ts[args.series_name], pad=4.0))
            except KeyError:
                y_vals.append(0.0)
                ax1.text(pos, 0.80-0.05*dwell_num, str(0.0)+ "/$\mu$s",
                color=TestProject.color_ts[args.series_name], size=12,
                bbox=dict(facecolor='white',
                edgecolor=TestProject.color_ts[args.series_name], pad=4.0))

        ax2.scatter(x_vals, y_vals, s=2)
        ax2.plot(x_vals, y_vals, alpha=0.5, linewidth=0.5)

    f1.set_size_inches(18.5, 5.5)
    #f1.savefig(TestProject.output_name+'_zhistogram_w_scandividers_dwellrange.pdf', dpi=200)
    f1.savefig(TestProject.output_name+'SOD_zhistogram_w_scandividers_dwellrange.pdf', dpi=200)
