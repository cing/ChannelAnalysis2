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
#from Transitions import mean_unique_macrostate_transition_counts
from PandasMD.Plot import plot_ts_by_resid_with_trans
from PandasMD.Transitions import macrostate_transitions_per_traj
from PandasMD.Coordination import macrostate_labels
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

        coord_df2_occprime = macrostate_labels(coord_df2_trim, drop_prime=True)
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

            trans_subset = trans[trans["TrajNum"]==traj]
            coord_subset = coord_df2_trim[coord_df2_trim["TrajNum"]==traj]

            for imagenum, t in enumerate(range(0,int(np.round(coord_subset["Time"].max())),2)):
                f1, ax = plt.subplots(1)

                coord_subset_less = coord_subset[coord_subset["Time"]<t]
                trans_subset_less = trans_subset[trans_subset["OccMacrostate0_Stop"]<t]

                plot_ts_by_resid_with_trans(ax, coord_subset_less, trans=trans_subset_less,
                                       trans_columns=trans_columns, state_colors=unique_state_colors,
                                       xlim=[0, 1000], ylim=[-8,10], skip=15,
                                       x_label="time (ns)", y_label="z (nm)", limit_state_draw=t)

                f1.gca().invert_yaxis()
                f1.set_size_inches(14.5, 5.5)
                t_leading='{num:05d}'.format(num=imagenum)
                f1.savefig(TestProject.output_name+"_ts_n"+str(traj)+"E_t"+str(t_leading)+".png", dpi=200)
