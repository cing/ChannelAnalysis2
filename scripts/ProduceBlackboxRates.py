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
from ChannelAnalysis2.Transitions import mode_coordstate_transitions_per_trajres
from ChannelAnalysis2.Coordination import coordination_labels, mode_coordination_labels
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
    '-c', dest='cfg_path', type=str, required=True,
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', type=str, nargs="+", required=True,
    help='column name of coordination data to compute rates')
    args = parser.parse_args()

    #f1, ax1 = plt.subplots()
    TestProject = Project(args.cfg_path, coord_ts=args.series_name)
    print("Successfully Loaded Project: %s" % TestProject.name)


    coord_colnames = ["S178","E177","L176","T175"]
    #coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name[0]],
    #                              TestProject.start_time,
    #                              TestProject.end_time)
    coord_df_noequil = TestProject.coord_ts[args.series_name[0]]
    coord_df_noequil["IonZRank"] = np.digitize(coord_df_noequil["Z"], bins=[-6, 6])

    trans = mode_coordstate_transitions_per_trajres(coord_df_noequil,
                                      states_per_transition=2, dwell_cut=2, transition_column="IonZRank",
                                      return_stats=False)

    forward_trans = trans[trans["IonZRank0_Start"] < trans["IonZRank1_Start"]]
    backward_trans = trans[trans["IonZRank0_Start"] > trans["IonZRank1_Start"]]

    total_ftransitions=0
    total_btransitions=0
    for traj_start, sub_trans in forward_trans.groupby(["TrajNum", "IonZRank0_Stop"]):
        prev_row = sub_trans.iloc[0]
        for idx, event in sub_trans.iterrows():
            curr_row = event
            if ((prev_row["IonZRank0_Start"] == 0 and prev_row["IonZRank1_Start"] == 1) and
               (curr_row["IonZRank0_Start"] == 1 and curr_row["IonZRank1_Start"] == 2) and
               (prev_row["IonZRank1_Stop"] < curr_row["IonZRank1_Stop"]+curr_row["IonZRank1_Dwell"])):
                    print("Traj "+str(prev_row["TrajNum"])+": Ion "+str(prev_row["ResidID"])+" entered at "+str(prev_row["IonZRank1_Stop"]))
                    print("Traj "+str(prev_row["TrajNum"])+": Ion "+str(curr_row["ResidID"])+" exited at "+str(curr_row["IonZRank1_Stop"]))
                    #print(prev_row)
                    #print(curr_row)
                    print("------")
                    total_ftransitions += 1.0
            prev_row = curr_row

    for traj_start, sub_trans in backward_trans.groupby(["TrajNum", "IonZRank0_Stop"]):
        prev_row = sub_trans.iloc[0]
        for idx, event in sub_trans.iterrows():
            curr_row = event
            if ((prev_row["IonZRank0_Start"] == 0 and prev_row["IonZRank1_Start"] == 1) and
               (curr_row["IonZRank0_Start"] == 1 and curr_row["IonZRank1_Start"] == 2) and
               (prev_row["IonZRank1_Stop"] < curr_row["IonZRank1_Stop"]+curr_row["IonZRank1_Dwell"])):
                    print("Traj "+str(prev_row["TrajNum"])+": Ion "+str(prev_row["ResidID"])+" entered at "+str(prev_row["IonZRank1_Stop"]))
                    print("Traj "+str(prev_row["TrajNum"])+": Ion "+str(curr_row["ResidID"])+" exited at "+str(curr_row["IonZRank1_Stop"]))
                    #print(prev_row)
                    #print(curr_row)
                    print("------")
                    total_btransitions += 1.0
            prev_row = curr_row

    total_steps = coord_df_noequil.groupby(["TrajNum"])["Time"].apply(lambda x: max(x.unique())).sum()
    print(total_ftransitions, total_btransitions, total_steps, total_ftransitions/total_steps, total_btransitions/total_steps)

    gg = trans.groupby(["TrajNum","Transition"])[gb_column].sum().reset_index()
    ggpivot = gg.pivot(index="Transition",columns="TrajNum",values=gb_column).fillna(0)
    ggstats = pd.concat([ggpivot.mean(axis=1),
                         ggpivot.std(axis=1),
                         ggpivot.sem(axis=1)], axis=1)

    ggstats.columns = ["Mean","STD","SEM"]

    print(ggstats)
    macrostate_size=coord_df2_merge.groupby(["ModeLabel"]).size()
    macrostate_percent=macrostate_size/sum(macrostate_size)
    print(macrostate_size)

    beforeafter=pd.Series(ggstats.index).str.split("-", return_type="frame")
    beforeafter.columns=["ModeLabel","NextModeLabel"]
    ggstats_beforeafter=ggstats.reset_index().join(beforeafter)
    ggstats_macro=ggstats_beforeafter[["Transition","Mean","SEM","ModeLabel"]].set_index("ModeLabel")

    ggstats_normed=ggstats_macro.join(pd.DataFrame(macrostate_percent, columns=["Percent"]))
    ggstats_normed["MeanPercent"]=ggstats_normed["Mean"]/ggstats_normed["Percent"]
    ggstats_normed["SEMPercent"]=ggstats_normed["SEM"]/ggstats_normed["Percent"]
    print(ggstats_normed)
    #print(ggstats_normed[ggstats_normed["Percent"]>0.05])

