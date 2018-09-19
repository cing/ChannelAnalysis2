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
from ChannelAnalysis2.Plot import plot_avgcoord_histogram
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
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-n', dest='traj_set', type=int, required=True,
    help='the trajset for the fileout suffix')
    args = parser.parse_args()

    coord_colnames = ["S178","E177","L176","T175","W"]
    colors = {"S178":"#43B649", "E177":"#F26122", "L176":"#5F60AB", "T175":"#F496BF", "W":"#20C4F4"}

    if len(args.cfg_path) == 1:
        # This builds a Project using the configuration file argument
        TestProject = Project(args.cfg_path, coord_ts=args.series_name)
        #TestProject = Project(args.cfg_path)
        print("Successfully Loaded Project: %s" % TestProject.name)

        for series_num, series_name in enumerate(args.series_name):
            # default = '1st Shell Na+ SF Coordination'
            coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                          TestProject.start_time,
                                          TestProject.end_time)

            coord_df2_trim = prune_column(coord_df_noequil, "Z", 5, 35) #FOR AN IC GATE CHECK!

            f1, ax = plt.subplots(1)
            plot_avgcoord_histogram(ax, coord_df2_trim,
                                     bins=150, xlim=[5,35], ylim=[4,6.0],
                                     x_label="position (Ang)",
                                     y_label="probability (arb. units)",
                                     colors=colors, line_style="solid",
                                     coord_colnames = coord_colnames, hist_vline=None)

        f1.set_size_inches(18.5, 2.75)
        #f1.savefig(TestProject.output_name+'_zhistogram_for_avgcoordlabels_tiltfix_long.pdf', dpi=200)
        f1.savefig(TestProject.output_name+'_zhistogram_for_avgcoordlabels_n'+str(args.traj_set)+'.pdf', dpi=200)
