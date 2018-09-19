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
from PandasMD.Coordination import coordination_labels, mode_coordination_labels, occupancy_populations_pertraj
from PandasMD.Coordination import OPLS_mode_coordination_labels
from PandasMD.Plot import plot_orderings
import matplotlib.pyplot as plt
import operator

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

    coord_colnames = ["S178","E177","L176","T175"]

    # We don't know if this is a competition dataset or not
    if args.series_name2:
        series_to_loop_over = [args.series_name, args.series_name2]
    else:
        series_to_loop_over = [args.series_name]
    print("We will loop over:", series_to_loop_over)

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path, coord_ts=[i for s in series_to_loop_over for i in s])
    print("Successfully Loaded Project: %s" % TestProject.name)


    all_series = []
    for series_id, series in enumerate(series_to_loop_over):

        coord_df_noequil = prune_time(TestProject.coord_ts[series[0]],
                                      TestProject.start_time,
                                      TestProject.end_time)
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14) 
        coord_df2_trim["CoordLabel"] = coordination_labels(coord_df2_trim, coord_colnames)

        if args.forcefield == "OPLS":
            coord_df2_merge = coord_df2_trim
            coord_df2_merge["ModeLabel"] = OPLS_mode_coordination_labels(coord_df2_merge)
        else:
            # For CHARMM we need get 2nd shell coordination, assumed to be the second series argument
            coord_df_noequil_2nd = prune_time(TestProject.coord_ts[series[1]],
                                          TestProject.start_time,
                                          TestProject.end_time)
            coord_df2_trim_2nd = prune_column(coord_df_noequil_2nd, "Z", -10, 14) 
            coord_df2_trim_2nd["CoordLabel_2nd"] = coordination_labels(coord_df2_trim_2nd, coord_colnames)

            coord_df2_merge=coord_df2_trim.merge(coord_df2_trim_2nd[["TrajNum","Time","ResidID","CoordLabel_2nd"]],
                               on=["TrajNum","Time","ResidID"])
            coord_df2_merge["ModeLabel"] = mode_coordination_labels(coord_df2_merge)

        all_series.append(coord_df2_merge)

    superdf = pd.concat(all_series)
    order_counts = defaultdict(int)
    coord_colvals = ["EC","S","E","EL","LT","CC"]
    print(coord_colvals)
    for name, group in superdf.groupby(["TrajNum","Time"]):
        temp_row = ["","","","","",""]
        for ion in group.iterrows():
            species = ion[1]["RowID"][0]
            lbl = ion[1]["ModeLabel"]
            if (lbl != "NONE") and (lbl != "EC"):
                if lbl == "L":
                    lbl = "EL"
                lbl_ind = coord_colvals.index(lbl)
                temp_row[lbl_ind] += species
        order_counts[tuple(temp_row)] += 1

    total_frames = np.sum(order_counts.values())
    for order, count in sorted(order_counts.items(), key=operator.itemgetter(1), reverse=True):
        print(order, float(count)/total_frames)

    fig, ax = plt.subplots(figsize=(20, 20))
    plot_orderings(ax, order_counts)
    #plt.subplots_adjust(left=0, right=1.0, bottom=0, top=1.0)
    plt.subplots_adjust(left=0, right=0.95, bottom=0, top=1.0)
    plt.axis('equal')
    plt.axis('off')
    plt.savefig(TestProject.output_name+"_macrostate_orderings2.pdf", dpi=200)
