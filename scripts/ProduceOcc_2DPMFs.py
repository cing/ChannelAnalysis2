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
from PandasMD.Coordination import mean_2d_pmf
from PandasMD.Plot import plot_macrostate_2d_pmfs
from matplotlib import gridspec
from PandasMD.Coordination import macrostate_labels, macrostate_occupancy_ts
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
    TestProject = Project(args.cfg_path, coord_ts=[args.series_name,])
    print("Successfully Loaded Project: %s" % TestProject.name)

    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name],
                                  TestProject.start_time,
                                  TestProject.end_time)

    coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
    #coord_df2_trim = prune_column(coord_df_noequil, "Z", -16, 16)
    #ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean() 
    #histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
    #ioncount_norm = 1
    #norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
    #print(norm_factor)
    # We're ditching the normalization at the moment...
    norm_factor = 1

    coord_df2_order = macrostate_labels(coord_df2_trim, drop_prime=True)
    #coord_df2_order1 = macrostate_occupancy_ts(coord_df2_order, occ=1, prime=False)
    coord_df2_order2 = macrostate_occupancy_ts(coord_df2_order, occ=2, prime=False)
    coord_df2_order3 = macrostate_occupancy_ts(coord_df2_order, occ=3, prime=False)

    # Do you want the single-ion PMF?
    if False:
        from PandasMD.Coordination import mean_axial_pmf
        from PandasMD.Plot import plot_pmf
        fig, ax = plt.subplots()
        pmf = mean_axial_pmf(coord_df2_order1[["Red","TrajNum"]], min_binval=2,
                              axis_column=["Red"], histrange=[-10,10], left_shift=False)
        plot_pmf(ax, pmf,
                 color="red",
                 xlim=[-10,10], ylim=[-1,5])
        fig.set_size_inches(12.5, 5.5)
        fig.savefig(TestProject.output_name+'_zhistogram_2d_for_occ1_tiltfix_np.pdf', dpi=200)

    # Plotting begins here
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 4, width_ratios=[3,1,3,1], height_ratios=[0.5,3])
    ax = plt.subplot(gs[1,0])
    axl = plt.subplot(gs[1,1], sharey=ax)
    axb = plt.subplot(gs[0,0], sharex=ax)
    plot_macrostate_2d_pmfs(ax, axl, axb, coord_df2_order2, axis_colors=["red","green"], hist_range=[-10,10,-10,10],
                            axis_columns=[["Red"],["Green"]],norm_factor=norm_factor,residual_pmfs=True)
    fig.set_size_inches(12.5, 5.5)
    fig.savefig(TestProject.output_name+'_zhistogram_2d_for_occ2_tiltfix_np.pdf', dpi=200)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 4, width_ratios=[3,1,3,1], height_ratios=[0.5,3])
    ax = plt.subplot(gs[1,0])
    axl = plt.subplot(gs[1,1], sharey=ax)
    axb = plt.subplot(gs[0,0], sharex=ax)
    plot_macrostate_2d_pmfs(ax, axl, axb, coord_df2_order3, axis_colors=["red","green"], hist_range=[-10,10,-10,10],
                            axis_columns=[["Red"],["Green"]],norm_factor=norm_factor,residual_pmfs=True)
    fig.set_size_inches(12.5, 5.5)
    fig.savefig(TestProject.output_name+'_zhistogram_2d_for_occ3_tiltfix_np.pdf', dpi=200)

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 4, width_ratios=[3,1,3,1], height_ratios=[0.5,3])
    ax = plt.subplot(gs[1,0])
    axl = plt.subplot(gs[1,1], sharey=ax)
    axb = plt.subplot(gs[0,0], sharex=ax)
    plot_macrostate_2d_pmfs(ax, axl, axb, coord_df2_order3, axis_colors=["green","blue"], hist_range=[-10,10,-10,10],
                            axis_columns=[["Green"],["Blue"]],norm_factor=norm_factor,residual_pmfs=True)
    fig.set_size_inches(12.5, 5.5)
    fig.savefig(TestProject.output_name+'_zhistogram_2d_for_occ3b_tiltfix_np.pdf', dpi=200)
    
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 4, width_ratios=[3,1,3,1], height_ratios=[0.5,3])
    ax = plt.subplot(gs[1,0])
    axl = plt.subplot(gs[1,1], sharey=ax)
    axb = plt.subplot(gs[0,0], sharex=ax)
    plot_macrostate_2d_pmfs(ax, axl, axb, coord_df2_order3, axis_colors=["red","blue"], hist_range=[-10,10,-10,10],
                            axis_columns=[["Red"],["Blue"]],norm_factor=norm_factor,residual_pmfs=True)
    fig.set_size_inches(12.5, 5.5)
    fig.savefig(TestProject.output_name+'_zhistogram_2d_for_occ3c_tiltfix_np.pdf', dpi=200)
