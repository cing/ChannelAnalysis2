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
#from Transitions import mean_unique_macrostate_transition_counts
from ChannelAnalysis2.Plot import plot_ts_by_resid_with_trans
from ChannelAnalysis2.Transitions import macrostate_transitions_per_traj
from ChannelAnalysis2.Coordination import macrostate_labels
import matplotlib.pyplot as plt
import matplotlib as mpl

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct
    RGB color.'''
    color_norm  = mpl.colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = mpl.cm.ScalarMappable(norm=color_norm, cmap='Set1')
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

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

def label_macrostate(row):
    if row["Prime"]:
        return str(row["Occupancy"]-1)+"'"
    else:
        return str(row["Occupancy"])

if __name__ == '__main__':
    parser = ArgumentParser(
    description='This script extracts basic transition statistics of \
    coordination with attention to statistics across multiple timeseries.')
    parser.add_argument(
    '-c', dest='cfg_path', type=str, required=True,
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    #f1, ax1 = plt.subplots()
    #TestProject = Project(args.cfg_path)
    TestProject = Project(args.cfg_path, coord_ts=args.series_name)
    print("Successfully Loaded Project: %s" % TestProject.name)

    fig = {}
    axes = {}

    for series_num, series_name in enumerate(args.series_name):
        coord_df_noequil = prune_time(TestProject.coord_ts[series_name],
                                      #TestProject.start_time,
                                      0,
                                      TestProject.end_time)
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)

        coord_df2_occprime = macrostate_labels(coord_df2_trim)
        coord_df2_occprime_low = coord_df2_occprime[coord_df2_occprime["Order"] == 0]


        coord_df2_macrostate = coord_df2_occprime_low.apply(label_macrostate, axis=1)
        coord_df2_occprime_low["OccMacrostate"] = coord_df2_macrostate

        states_per_transition=2
        transition_column="OccMacrostate"

        trans = macrostate_transitions_per_traj(coord_df2_occprime_low[["Frame","OccMacrostate","TrajNum"]],
                                        transition_column=transition_column,
                                        dwell_cut=150, return_stats=False, 
                                        states_per_transition=states_per_transition)

        trans_columns=[]
        for statenum in range(states_per_transition):
            trans_columns.append(transition_column+str(statenum)+"_Start")
            #trans[transition_column+str(statenum)+"_Start"] *= TestProject.dt
            trans[transition_column+str(statenum)+"_Stop"] *= TestProject.dt
            trans[transition_column+str(statenum)+"_Dwell"] *= TestProject.dt

        #print(trans)
        #trans = macrostate_transitions_per_traj(coord_df2_occprime_low[["Frame","OccMacrostate","TrajNum"]],
        #                                transition_column="OccMacrostate",
        #                                dwell_cut=5, return_stats=True, states_per_transition=3)


        # This depends on your system!
        #all_possible_states=['0',"0'",'1',"1'",'2', "2'", '3', "3'",'4',"4'"]
        all_possible_states=['0','1','2','3','4',"0'","1'","2'","3'","4'"]
        cmap = get_cmap(len(all_possible_states))
        unique_state_colors = {state: cmap(ind) for ind, state in enumerate(all_possible_states)}

        for traj in coord_df2_trim["TrajNum"].unique():

            if traj not in fig:
                f1, ax = plt.subplots(len(args.series_name))
                fig[traj] = f1
                axes[traj] = ax

            target_ax = axes[traj]

            trans_subset = trans[trans["TrajNum"]==traj]
            coord_subset = coord_df2_trim[coord_df2_trim["TrajNum"]==traj]

            #import pdb
            #pdb.set_trace()
            plot_ts_by_resid_with_trans(target_ax, coord_subset, trans=trans_subset,
                                   trans_columns=trans_columns, state_colors=unique_state_colors,
                                   #xlim=[0, 1000], ylim=[-10,14],
                                   xlim=[0, 1000], ylim=[-8,10], skip=15,
                                   x_label="time (ns)", y_label="axial position (nm)")

    for traj,f1 in fig.iteritems():
        f1.gca().invert_yaxis()
        f1.set_size_inches(18.5, 5.5*len(args.series_name))
        f1.savefig(TestProject.output_name+"_ts_n"+str(traj)+"C.pdf", dpi=200)
        #f1.savefig(TestProject.output_name+"_ts_n"+str(traj)+"D.pdf", dpi=200)
