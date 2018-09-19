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
from PandasMD.Coordination import mean_axial_pmf
from PandasMD.Plot import plot_pmf
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
    '-c', dest='cfg_path', nargs="+", type=str, required=True,
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', nargs="+", type=str, required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-fname', dest='file_name', type=str, required=False,
    help='file name prefix for output')
    args = parser.parse_args()

    f1, ax = plt.subplots(1)
    target_ax = ax

    if len(args.cfg_path) == 1:
        # This builds a Project using the configuration file argument
        TestProject = Project(args.cfg_path, coord_ts=args.series_name)
        #TestProject = Project(args.cfg_path)
        print("Successfully Loaded Project: %s" % TestProject.name)

        for series_name in args.series_name:
            coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                          TestProject.start_time,
                                          TestProject.end_time)

            # Extract the mean number of ions in the range -2.0 to 2.0
            coord_df2_trim = prune_column(coord_df_noequil, "Z", -20, 14)
            pmf = mean_axial_pmf(coord_df2_trim, histrange=[-20,14])

            plot_pmf(target_ax, pmf,
                     color=TestProject.color_ts[series_name],
                     #xlim=[-10,14], ylim=[-5,5],
                     #xlim=[-10,14], ylim=[-2,3],
                     #xlim=[-20,14], ylim=[-2,3],
                     xlim=[-10,10], ylim=[-2,3],
                     x_label="position (Ang)",
                     y_label="free energy (kca/mol)")

    else:
        for cfg, series_name in zip(args.cfg_path,args.series_name):
            # This builds a Project using the configuration file argument
            # TestProject = Project(cfg)
            TestProject = Project(cfg, coord_ts=[series_name,])
            print("Successfully Loaded Project: %s" % TestProject.name)

            # default = '1st Shell Na+ SF Coordination'
            coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                          TestProject.start_time,
                                          TestProject.end_time)

            # Extract the mean number of ions in the range -2.0 to 2.0
            #coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
            coord_df2_trim = prune_column(coord_df_noequil, "Z", -20, 14)
            pmf = mean_axial_pmf(coord_df2_trim, histrange=[-20,14])
            #coord_df2_trim = prune_column(coord_df_noequil, "Z", -14, 14)
            #pmf = mean_axial_pmf(coord_df2_trim, histrange=[-14,14])

            plot_pmf(target_ax, pmf,
                     color=TestProject.color_ts[series_name],
                     xlim=[-20,14], ylim=[-2,3],
                     #xlim=[-10,14], ylim=[-2,3],
                     #xlim=[-10,10], ylim=[-2,3],
                     #xlim=[-10,14], ylim=[-5,5],
                     x_label="position (Ang)",
                     y_label="free energy (kca/mol)")

    f1.set_size_inches(18.5, 5.5)
    if args.file_name != None:
        f1.savefig(args.file_name+'_zpmf_bulk2_short.pdf', dpi=200)
    else:
        f1.savefig(TestProject.output_name+'_zpmf_bulk_tiltfix_longest.pdf', dpi=200)
