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
from PandasMD.Coordination import coordination_labels, OPLS_mode_coordination_labels, occupancy_populations_pertraj, mode_coordination_labels
from PandasMD.Coordination import dunking_labels
from PandasMD.Plot import plot_mode_occbarchart
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
    '-ts2', dest='series_name2', type=str, nargs="+", default=[],
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-ff', dest='forcefield', type=str, default="CHARMM",
    help='forcefield dictates how we define states')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded %s Project: %s" % (args.forcefield, TestProject.name))

    coord_colnames = ["S178","E177","L176","T175"]

    # We don't know if this is a competition dataset or not
    if args.series_name2:
        series_to_loop_over = [args.series_name, args.series_name2]
    else:
        series_to_loop_over = [args.series_name]
    print("We will loop over:", series_to_loop_over)

    coord_colvals = ["EC","S","E","EL","LT","CC"]
    f1, axes = plt.subplots(1,len(coord_colvals)+3, sharey=True)

    # This is a dictionary that holds the previous bar chart y values
    # This will get populated after series_id > 0
    yoffset={}
    ptitles={}

    # Compute dunking statistics across the dataset
    dihe_df2_merge = prune_time(TestProject.dihe_ts["E177 Chi1-Chi2 Dihedral Angles"],
                                TestProject.start_time,
                                TestProject.end_time)
    dunk_labels = dunking_labels(dihe_df2_merge,
                   ["S1-Chi2", 'S2-Chi2', 'S3-Chi2', 'S4-Chi2'])
    dihe_df2_merge = dihe_df2_merge.join(dunk_labels)

    dunk_per_traj={}
    for traj, data in dihe_df2_merge.groupby("TrajNum"):
        z = data["SALL-dunked"].value_counts()
        dunk_per_traj[traj] = z/z.sum()

    dunk_df = pd.DataFrame(dunk_per_traj).fillna(0)

    traj_df = pd.DataFrame(dunk_per_traj).fillna(0)
    traj_mean = dunk_df.mean(axis=1).sort_index()
    traj_sem= dunk_df.sem(axis=1).sort_index()
    z=pd.concat([traj_mean, traj_sem],axis=1)
    z.columns=["Mean","SEM"]
    mean_across_traj = np.sum(traj_mean*(traj_mean.index))
    sem_average = np.sqrt(np.sum(np.power(traj_sem,2)))
    ptitle = str(np.around(mean_across_traj, decimals=1))+" ("+str(np.around(sem_average, decimals=1))+")"
    xvals = z.index
    plot_mode_occbarchart(axes.flatten()[-1], xvals, traj_mean, traj_sem,
                      ptitle=ptitle,
                      xlim=[0,5], ylim=[0,1.0])
    print("Dunk")
    print(z)
    print(str(mean_across_traj)+" "+str(sem_average))
