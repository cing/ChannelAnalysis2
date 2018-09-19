""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted.

# Example script usage using Project configuration file (here text output is useful for writing values in the text):
python ${script} -ff CHARMM -c test2.cfg  -ts "1st Shell K+ SF Coordination" "2nd Shell K+ SF Coordination" > NAVAB_XTAL_PD-NORES-0kcal-CHARMM_NONB_WT_POT150_J_modeoccupancy.txt
python ${script} -ff CHARMM -c test3.cfg  -ts "1st Shell Na+ SF Coordination" "2nd Shell Na+ SF Coordination" -ts2 "1st Shell K+ SF Coordination" "2nd Shell K+ SF Coordination" > NAVAB_XTAL_PD-NORES-0kcal-CHARMM_NONB_WT_SOD150POT150_J_modeoccupancy.txt
"""

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from ChannelAnalysis2.Initialize import Project
import itertools
from collections import defaultdict
from ChannelAnalysis2.Coordination import coordination_labels, OPLS_mode_coordination_labels, occupancy_populations_pertraj, mode_coordination_labels
from ChannelAnalysis2.Coordination import dunking_labels
from ChannelAnalysis2.Coordination import joint_occupancy_populations_pertraj
from ChannelAnalysis2.Plot import plot_mode_occbarchart
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
    #all_coord = args.series_name + args.series_name2 + ["E177 Chi1-Chi2 Dihedral Angles"]
    TestProject = Project(args.cfg_path,) #coord_ts=all_coord)
    print("Successfully Loaded %s Project: %s" % (args.forcefield, TestProject.name))

    coord_colnames = ["S178","E177","L176","T175"]

    # We don't know if this is a competition dataset or not
    if args.series_name2:
        series_to_loop_over = [args.series_name, args.series_name2]
    else:
        series_to_loop_over = [args.series_name]
    print("We will loop over:", series_to_loop_over)

    # WARNING FOR S178A, and S178A/S180A mutants only!
    #coord_colvals = ["EC","E","EL","LT","CC"]
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
    # No matter what, get the labels right, later we can do a subset
    chi1_cols = ["S1-Chi1", 'S2-Chi1', 'S3-Chi1', 'S4-Chi1']
    chi2_cols = ["S1-Chi2", 'S2-Chi2', 'S3-Chi2', 'S4-Chi2']

    dunk_labels = dunking_labels(dihe_df2_merge, chi2_cols,
                                 dihe_colnames2=chi1_cols,)
    dihe_df2_merge = dihe_df2_merge.join(dunk_labels)

    dunk_per_traj={}
    for traj, data in dihe_df2_merge.groupby("TrajNum"):
        z = data["SALL-dunked"].value_counts()
        dunk_per_traj[traj] = z/z.sum()

    dunk_df = pd.DataFrame(dunk_per_traj).fillna(0)
    # End dunking statistics calculation

    all_series = []
    for series_id, series in enumerate(series_to_loop_over):

        coord_df_noequil = prune_time(TestProject.coord_ts[series[0]],
                                      TestProject.start_time,
                                      TestProject.end_time)
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)

        if "E177p" in coord_df2_trim.columns:
            coord_df2_trim["E177"] += coord_df2_trim["E177p"]

        coord_df2_trim["CoordLabel"] = coordination_labels(coord_df2_trim, coord_colnames)

        if series_id == 0:
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

    pops = joint_occupancy_populations_pertraj(all_series,
                                               coord_col="ModeLabel",
                                               #WARNING
                                               #coord_sfcolvals=['E','EL','LT'],
                                               coord_sfcolvals=['S','E','EL','LT'],
                                               coord_colvals=coord_colvals)


    max_index=3
    yscaling=1
    for ax, mode in zip(axes.flatten()[:-1], ["CHAN","SF"]+coord_colvals):
        trajdata = pops[mode]
        all_mean = trajdata.mean(axis=1).sort_index()
        all_sem = trajdata.sem(axis=1).sort_index()
        for series_id in range(len(series_to_loop_over)):
            traj_mean = all_mean[:,series_id]
            traj_sem = all_sem[:,series_id]
            z=pd.concat([traj_mean, traj_sem],axis=1)
            z.columns=["Mean","SEM"]
            xvals = z.index
            mean_across_traj = np.sum(traj_mean*(traj_mean.index))
            sem_average = np.sqrt(np.sum(np.power(traj_sem,2)))
            print("Series: "+str(series_id)+", Mode: "+mode+" = "+str(mean_across_traj)+" "+str(sem_average))
            print(z)

            # The first time through, we set the yoffset vector to zero
            if series_id == 0:
                yoffset[mode] = pd.DataFrame(np.zeros([len(traj_mean),2]), index=xvals, columns=["Mean","SEM"])

            # I'm handling 3 cases, the loop is over 1 series, the loop is on the first loop of 2 series, and 
            # the loop is on the second loop of 2 series
            if len(series_to_loop_over) > 1:
                if series_id == 0:
                    ptitle = None
                else:
                    ptitle = str(np.around(mean_across_traj, decimals=1))+" ("+str(np.around(sem_average, decimals=1))+")"
                    ptitle = ptitles[mode] + ", " + ptitle
            else:
                yscaling = 1.
                ptitle = str(np.around(mean_across_traj, decimals=1))+" ("+str(np.around(sem_average, decimals=1))+")"

            if mode == "CHAN":
                # Chan starts with 1, like, unanimously! Make sure to doublecheck tho...
                plot_mode_occbarchart(ax, xvals, yscaling*traj_mean, yscaling*traj_sem,
                                  ptitle=ptitle, yoffset=yoffset[mode].loc[z.index,:]["Mean"],
                                  xlim=[1,4], ylim=[0,1.0])
            elif mode == "SF":
                plot_mode_occbarchart(ax, xvals, yscaling*traj_mean, yscaling*traj_sem,
                                  ptitle=ptitle, yoffset=yoffset[mode].loc[z.index,:]["Mean"],
                                  #xlim=[0,4], ylim=[0,1.0])
                                  xlim=[1,4], ylim=[0,1.0])
            else:
                plot_mode_occbarchart(ax, xvals, yscaling*traj_mean, yscaling*traj_sem,
                                  ptitle=ptitle, yoffset=yoffset[mode].loc[z.index,:]["Mean"],
                                  xlim=[0,max_index], ylim=[0,1.0])

            # After we plot it, we increment the yoffsets for the second plotting series,
            # the reason we do fillna is because we may not have the right alignment...
            if series_id == 0:
                yoffset[mode] = (yoffset[mode]+z*yscaling).fillna(0)
                ptitle = str(np.around(mean_across_traj, decimals=1))+" ("+str(np.around(sem_average, decimals=1))+")"
                ptitles[mode] = ptitle

            #print("Done Loop")
            #print(yoffset)

    #f1.set_size_inches(5.5, 0.75)
    f1.set_size_inches(11, 1.5)
    #f1.savefig(TestProject.output_name+'_POT_occupancy_for_modelabels.pdf', dpi=200)
    f1.savefig(TestProject.output_name+'_occupancy_for_modelabels_Joint2.pdf', dpi=200)
