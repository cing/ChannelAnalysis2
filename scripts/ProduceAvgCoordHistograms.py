""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted.

# Example script usage using Project configuration file:
python ${script} -c test2.cfg  -ts "1st Shell K+ SF Oxy Coordination"
python ${script} -c test3.cfg  -ts "1st Shell Na+ SF Oxy Coordination" "1st Shell K+ SF Oxy Coordination"
"""

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
    args = parser.parse_args()

    coord_colnames = ["S178","E177","L176","T175","W"]
    colors = {"S178":"#43B649", "E177":"#F26122", "L176":"#5F60AB", "T175":"#F496BF", "W":"#20C4F4"}

    f1, ax = plt.subplots(1)

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

            # Extract the mean number of ions in the range -2.0 to 2.0
            coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 10)
            #coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
            #coord_df2_trim = prune_column(coord_df_noequil, "Z", 10, 40) #FOR AN IC GATE CHECK!

            if "E177p" in coord_df2_trim.columns:
                coord_df2_trim["E177"] += coord_df2_trim["E177p"]

            if series_num == 0:
                line_style = "solid"
            else:
                line_style = "dashed"

            #plot_avgcoord_histogram(ax, coord_df2_trim, bins=300, xlim=[-10,8], ylim=[0,8.0],
            plot_avgcoord_histogram(ax, coord_df2_trim, bins=300, xlim=[-10,10], ylim=[0,8.0],
            #plot_avgcoord_histogram(ax, coord_df2_trim, bins=300, xlim=[10,40], ylim=[0,6.0],
            #plot_avgcoord_histogram(ax, coord_df2_trim, bins=300, xlim=[-10,8], ylim=[0,18.0],
                                     x_label="position (Ang)",
                                     y_label="probability (arb. units)",
                                     colors=colors, line_style=line_style,
                                     coord_colnames = coord_colnames, hist_vline=None)
    else:
        for series_num, (cfg, series_name) in enumerate(zip(args.cfg_path,args.series_name)):
            TestProject = Project(cfg, coord_ts=[series_name,])
            print("Successfully Loaded Project: %s" % TestProject.name)

            coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                          TestProject.start_time,
                                          TestProject.end_time)
            #coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 8)
            coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 10)

            if "E177p" in coord_df2_trim.columns:
                coord_df2_trim["E177"] += coord_df2_trim["E177p"]

            if series_num == 0:
                line_style = "solid"
            else:
                line_style = "dashed"
            #plot_avgcoord_histogram(ax, coord_df2_trim, bins=300, xlim=[-10,8], ylim=[0,8.0],
            plot_avgcoord_histogram(ax, coord_df2_trim, bins=300, xlim=[-10,10], ylim=[0,8.0],
                                     x_label="position (Ang)",
                                     y_label="probability (arb. units)",
                                     colors=colors, line_style=line_style,
                                     coord_colnames = coord_colnames, hist_vline=None)

    f1.set_size_inches(18.5, 2.75)
    f1.savefig(TestProject.output_name+'_zhistogram_for_avgcoordlabels_tiltfix_long.pdf', dpi=200)
