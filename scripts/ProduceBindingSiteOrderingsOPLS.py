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
from Coordination import coordination_labels, OPLS_mode_coordination_labels, occupancy_populations_pertraj
from Plot import plot_mode_occbarchart
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
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-ts2', dest='series_name2', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    coord_colnames = ["S178","E177","L176","T175"]
    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name[0]],
                                  TestProject.start_time,
                                  TestProject.end_time)
    coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
    #print(coord_df2_trim.shape)

    series_dfs = []
    for series_num, series_name in enumerate(args.series_name):
        if series_num == 0:
            coord_df2_trim["CoordLabel"] = coordination_labels(coord_df2_trim, coord_colnames)
            coord_df2_merge = coord_df2_trim
            
        #else:
        #    coord_df_noequil_2nd = prune_time(TestProject.coord_ts[series_name],
        #                                  TestProject.start_time,
        #                                  TestProject.end_time)
        #    coord_df2_trim_2nd = prune_column(coord_df_noequil_2nd, "Z", -10, 14)
        #    coord_df2_trim_2nd["CoordLabel_2nd"] = coordination_labels(coord_df2_trim_2nd, coord_colnames)

    #coord_df2_merge=coord_df2_trim.merge(coord_df2_trim_2nd[["TrajNum","Time","ResidID","CoordLabel_2nd"]],
    #                       on=["TrajNum","Time","ResidID"])

    coord_df2_merge["ModeLabel"] = OPLS_mode_coordination_labels(coord_df2_merge)
    #coord_df2_merge.to_pickle("test.df")
    #coord_df2_merge=pd.read_pickle("test.df")

    superdf = coord_df2_merge
    order_counts = defaultdict(int)
    coord_colvals = ["EC","S","E","EL","LT","CC"]
    print(coord_colvals)
    for name, group in superdf.groupby(["TrajNum","Time"]):
        temp_row = ["","","","","",""]
        for ion in group.iterrows():
            species = ion[1]["RowID"][0]
            lbl = ion[1]["ModeLabel"]
            if (lbl != "NONE") and (lbl != "EC"):
                lbl_ind = coord_colvals.index(lbl)
                temp_row[lbl_ind] += species
        order_counts[tuple(temp_row)] += 1

    total_frames = np.sum(order_counts.values())
    import operator
    for order, count in sorted(order_counts.items(), key=operator.itemgetter(1)):
        print(order, float(count)/total_frames)

    #coord_df2_merge.to_pickle("test.df")
    #coord_df2_merge=pd.read_pickle("test.df")


    #f1, axes = plt.subplots(len(coord_colvals), sharex=True)

    #pops = occupancy_populations_pertraj(coord_df2_merge, coord_col="ModeLabel",
    #                              coord_colvals=coord_colvals)

    #max_index=3
    #for ax, mode in zip(axes, coord_colvals):
    #    trajdata = pops[mode]
    #    print(mode)
    #    traj_mean = trajdata.mean(axis=1).sort_index()
    #    traj_sem = trajdata.sem(axis=1).sort_index()
    #    print(traj_mean)
    #    print(np.sum(traj_mean*(traj_mean.index)))

    #    plot_mode_occbarchart(ax, traj_mean, traj_sem,
    #                          xlim=[0,max_index], ylim=[0,1.0])

    #f1.set_size_inches(5.5, 5.5)
    ##f1.savefig('SOD150_NONB_occupancy_for_modelabels.pdf', dpi=200)
    #f1.savefig(TestProject.output_name+'_SOD_occupancy_for_modelabels.pdf', dpi=200)
