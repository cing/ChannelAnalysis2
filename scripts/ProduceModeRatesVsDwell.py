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
from PandasMD.Transitions import mode_coordstate_transitions_per_trajres
from PandasMD.Coordination import coordination_labels, mode_coordination_labels
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
    coord_df_noequil = prune_time(TestProject.coord_ts[args.series_name[0]],
                                  TestProject.start_time,
                                  TestProject.end_time)
    #coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
    coord_df2_trim = coord_df_noequil
    ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean()
    histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
    norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
    print(norm_factor)

    for series_num, series_name in enumerate(args.series_name):
        if series_num == 0:
            coord_df2_trim["CoordLabel"] = coordination_labels(coord_df2_trim, coord_colnames)
        else:
            coord_df_noequil_2nd = prune_time(TestProject.coord_ts[series_name],
                                          TestProject.start_time,
                                          TestProject.end_time)
            #coord_df2_trim_2nd = prune_column(coord_df_noequil_2nd, "Z", -10, 14)
            coord_df2_trim_2nd = coord_df_noequil_2nd
            coord_df2_trim_2nd["CoordLabel_2nd"] = coordination_labels(coord_df2_trim_2nd, coord_colnames)
#

    coord_df2_merge=coord_df2_trim.merge(coord_df2_trim_2nd[["TrajNum","Time","ResidID","CoordLabel_2nd"]],
                           on=["TrajNum","Time","ResidID"])

    f1, ax = plt.subplots()
    coord_df2_merge["ModeLabel"] = mode_coordination_labels(coord_df2_merge)

    trans = mode_coordstate_transitions_per_trajres(coord_df2_merge,
                                      dwell_cut=2, transition_column="ModeLabel",
                                      #dwell_cut=5, transition_column="ModeLabel",
                                      return_stats=True)

    compute_rates = True

    # Compute total number of frames using the last recorded frame per trajectory
    if compute_rates:
        total_steps = coord_df2_merge.groupby(["TrajNum"])["Time"].apply(lambda x: max(x.unique()))
        trans_normalized = trans.set_index("TrajNum")
        # Factor of 1000 puts things in units of microsecond.
        trans_normalized["Rate"] = 1000*trans_normalized["Count"].div(total_steps,
                                                                 axis="index").dropna()
        trans = trans_normalized.reset_index()
        gb_column="Rate"
    else:
        gb_column="Count"

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

