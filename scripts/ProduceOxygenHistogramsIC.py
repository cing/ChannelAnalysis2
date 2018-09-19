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
    '-n', dest='traj_set', type=int, required=True,
    help='the trajset for the fileout suffix')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    frames = 1
    f1, ax = plt.subplots(frames)

    for res_ts, res_df in TestProject.resid_ts.iteritems():
        subunit_dfs = []
        for subunit in ["S1","S2","S3","S4"]:
            temp_df = res_df[["Time","Frame","TrajNum",subunit+"-X",subunit+"-Y",subunit+"-Z","RowID"]].copy()
            temp_df.columns = ["Time","Frame","TrajNum","X","Y","Z","RowID"]
            temp_df["Subunit"]=subunit
            subunit_dfs.append(temp_df)

        res_df_reshape = pd.concat(subunit_dfs, ignore_index=True)

        plot_histogram_from_ts(ax, res_df_reshape["Z"], bins=150, xlim=[5,35],
                               color=TestProject.color_ts[res_ts],
                               ylim=[0,4.0], hist_range=[5,35],
                               x_label="position (Ang)",
                               y_label="probability (arb. units)",
                               norm_factor=10/res_df_reshape.shape[0], fill=True)

    f1.set_size_inches(12.5, 5.5*frames)
    f1.savefig(TestProject.output_name+'_oxygen_distribution_n'+str(args.traj_set)+'.pdf', dpi=200)
