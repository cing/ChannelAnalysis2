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
from PandasMD.Coordination import coordination_labels, mode_coordination_labels
from PandasMD.Coordination import OPLS_mode_coordination_labels
from PandasMD.Plot import plot_mode_histogram
from PandasMD.Transitions import macrostate_transitions_per_traj
from PandasMD.Coordination import macrostate_labels, macrostate_occupancy_ts
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')

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
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-ts2', dest='series_name2', type=str, nargs="+", default=[],
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-ff', dest='forcefield', type=str, default="CHARMM",
    help='forcefield dictates how we define states')
    args = parser.parse_args()

    # We don't know if this is a competition dataset or not
    if args.series_name2:
        series_to_loop_over = [args.series_name, args.series_name2]
    else:
        series_to_loop_over = [args.series_name]
    print("We will loop over:", series_to_loop_over)

    ## This builds a Project using the configuration file argument
    all_coord = args.series_name + args.series_name2
    TestProject = Project(args.cfg_path, coord_ts=all_coord)

    print("Successfully Loaded Project: %s" % TestProject.name)

    coord_colnames = ["S178","E177","L176","T175"]

    for series_id, series in enumerate(series_to_loop_over):

        coord_df_noequil = prune_time(TestProject.coord_ts[series[0]],
                                      TestProject.start_time,
                                      TestProject.end_time)
                                      #50,
                                      #150)

        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
        if "E177p" in coord_df2_trim.columns:
            coord_df2_trim["E177"] += coord_df2_trim["E177p"]

        coord_df_color = TestProject.color_ts[series[0]]
        coord_df2_trim["CoordLabel"] = coordination_labels(coord_df2_trim, coord_colnames)
        #ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean()
        #histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
        #norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
        #print(norm_factor)

        if args.forcefield == "OPLS":
            coord_df2_merge = coord_df2_trim
            coord_df2_merge["ModeLabel"] = OPLS_mode_coordination_labels(coord_df2_merge)
        else:
            # For CHARMM we need get 2nd shell coordination, assumed to be the second series argument
            coord_df_noequil_2nd = prune_time(TestProject.coord_ts[series[1]],
                                          TestProject.start_time,
                                          TestProject.end_time)
                                          #50, 150)
            coord_df2_trim_2nd = prune_column(coord_df_noequil_2nd, "Z", -10, 14)
            if "E177p" in coord_df2_trim_2nd.columns:
                coord_df2_trim_2nd["E177"] += coord_df2_trim_2nd["E177p"]
            coord_df2_trim_2nd["CoordLabel_2nd"] = coordination_labels(coord_df2_trim_2nd, coord_colnames)

            coord_df2_merge=coord_df2_trim.merge(coord_df2_trim_2nd[["TrajNum","Time","ResidID","CoordLabel_2nd"]],
                               on=["TrajNum","Time","ResidID"])
            coord_df2_merge["ModeLabel"] = mode_coordination_labels(coord_df2_merge)

        coord_df2_order = macrostate_labels(coord_df2_merge, drop_prime=True)

        f1, (ax2,ax3) = plt.subplots(2)

        # FOR CHARMM
        #plot_mode_histogram(ax, coord_df2_merge, xlim=[-10,8], ylim=[0,1.2],
        # FOR OPLS
        #plot_mode_histogram(ax, coord_df2_merge, xlim=[-8,10], ylim=[0,1.2],

        plot_df = coord_df2_order[coord_df2_order["Occupancy"]==2]
        print(len(plot_df))
        histogram, edges = np.histogram(plot_df["Z"], bins=300, range=[-8,10], normed=False)
        #norm_factor = (2.0*len(plot_df))/(sum(histogram)*(edges[1]-edges[0]))
        norm_factor = (0.65*2.0)/(sum(histogram)*(edges[1]-edges[0]))
        plot_mode_histogram(ax2, plot_df, xlim=[-8,10], ylim=[0,1.2],
                             x_label="position (Ang)",
                             y_label="probability (arb. units)",
                             norm_factor=norm_factor, sumdist_color=coord_df_color)

        plot_df = coord_df2_order[coord_df2_order["Occupancy"]==3]
        print(len(plot_df))
        histogram, edges = np.histogram(plot_df["Z"], bins=300, range=[-8,10], normed=False)
        #norm_factor = (3.0*len(plot_df))/(sum(histogram)*(edges[1]-edges[0]))
        norm_factor = (0.65*3.0)/(sum(histogram)*(edges[1]-edges[0]))
        plot_mode_histogram(ax3, plot_df, xlim=[-8,10], ylim=[0,1.2],
                             x_label="position (Ang)",
                             y_label="probability (arb. units)",
                             norm_factor=norm_factor, sumdist_color=coord_df_color)

        f1.set_size_inches(18.5, 5.5)
        f1.savefig(TestProject.output_name+'_zhistogram_for_modelabels_occsplit_tiltfix_short.pdf', dpi=200)