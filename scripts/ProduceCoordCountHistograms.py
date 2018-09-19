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
from PandasMD.Plot import plot_coordcount_histogram
import matplotlib.pyplot as plt

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
    TestProject = Project(args.cfg_path, coord_ts=[args.series_name])
    print("Successfully Loaded Project: %s" % TestProject.name)

    coord_colnames = ["S178","E177","L176","W"]
    colors = {"S178":"#43B649", "E177":"#F26122", "L176":"#5F60AB", "T175":"#F496BF", "W":"#20C4F4"}

    f1, axes = plt.subplots(len(coord_colnames))

    for coord_num, (ax, coord_name) in enumerate(zip(axes, coord_colnames)):
        # default = '1st Shell Na+ SF Coordination'
        coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name],
                                      TestProject.start_time,
                                      TestProject.end_time)

        # Extract the mean number of ions in the range -2.0 to 2.0
        #coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 8)
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 10)

        ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean() 
        histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
        norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))

        plot_coordcount_histogram(ax, coord_df2_trim, bins=300, xlim=[-10,10], ylim=[0,0.5],
                                  x_label="position (Ang)",
                                  y_label="probability (arb. units)",
                                  coord_min=1,
                                  coord_colname=coord_name, norm_factor=norm_factor,
                                  hist_vline=None)

    f1.set_size_inches(18.5, 18.5)
    f1.savefig(TestProject.output_name+'_zhistogram_for_coordcounts.pdf', dpi=200)
