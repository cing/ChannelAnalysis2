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
from Plot import plot_order_histograms, plot_histogram_from_ts
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
    '-ts', dest='series_name', type=str, required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    frames = 4
    f1, ax = plt.subplots(frames)

    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name],
                                  TestProject.start_time,
                                  TestProject.end_time)

    coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
    ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean() 
    histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
    norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
    print(norm_factor)

    for res_ts, res_df in TestProject.resid_ts.iteritems():
        subunit_dfs = []
        for subunit in ["A","B","C","D"]:
            temp_df = res_df[["Time","Frame","TrajNum",subunit+"-X",subunit+"-Y",subunit+"-Z","RowID"]].copy()
            temp_df.columns = ["Time","Frame","TrajNum","X","Y","Z","RowID"]
            temp_df["Subunit"]=subunit
            subunit_dfs.append(temp_df)

        res_df_reshape = pd.concat(subunit_dfs, ignore_index=True)

        plot_histogram_from_ts(ax[0], res_df_reshape["Z"], bins=300, xlim=[-8,10],
                               color=TestProject.color_ts[res_ts],
                               ylim=[0,1.0], hist_range=[-10,14],
                               x_label="position (Ang)",
                               y_label="probability (arb. units)",
                               norm_factor=10/res_df_reshape.shape[0], fill=True)

    plot_histogram_from_ts(ax[0], coord_df2_trim["Z"], bins=300, xlim=[-8,10],
                           color="#000000",
                           ylim=[0,1.0],
                           x_label="position (Ang)",
                           y_label="probability (arb. units)",
                           norm_factor=norm_factor)

    plot_order_histograms(ax[1:], coord_df2_trim, xlim=[-8,10], ylim=[0,0.7],
                             x_label="position (Ang)",
                             y_label="probability (arb. units)",
                             norm_factor=norm_factor)

    f1.set_size_inches(12.5, 5.5*frames)
    f1.savefig(TestProject.output_name+'_zhistogram_for_ordered_occ.pdf', dpi=200)
