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
from PandasMD.Plot import plot_mode_histogram
from PandasMD.Coordination import dunking_labels
import matplotlib.pyplot as plt

def prune_time(df, start, end):
    return df[(df["Time"] >= start) & (df["Time"] <= end)]

def prune_column(df, col, start, end):
    return df[(df[col] >= start) & (df[col] <= end)]

# There are many ways to implement this. This takes like half
# an hour to run, but... it's not bad!
# Returns 1 if coordinated ion has dunked GLU, 0 is coordinated ion does not
# Returns 2 if coordinated ion has BOTH dunked and undunked GLU in coord. shell
def coord_glustate(row):
    idxs = row[row.str.contains(str(row["ResidID"])) == True].index
    dunk_vals = [row["S"+str(chain[1])+"-dunked"] for chain in idxs]
    if len(set(dunk_vals)) <= 1:
        return dunk_vals[0]
    else:
        return 2

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
    args = parser.parse_args()

    ## This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)

    # We don't know if this is a competition dataset or not
    if args.series_name2:
        series_to_loop_over = [args.series_name, args.series_name2]
    else:
        series_to_loop_over = [args.series_name]
    print("We will loop over:", series_to_loop_over)

    for series_id, series in enumerate(series_to_loop_over):

        coord_df_noequil = prune_time(TestProject.coord_ts[series[0]],
                                      TestProject.start_time,
                                      TestProject.end_time)
                                      #50,
                                      #150)

        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 14)
        coord_df_color = TestProject.color_ts[series[0]]
        ioncount_norm = coord_df2_trim.groupby(["TrajNum","Time"]).size().mean()
        histogram, edges = np.histogram(coord_df2_trim["Z"], bins=300, range=[-10,14], normed=False)
        norm_factor = ioncount_norm/(sum(histogram)*(edges[1]-edges[0]))
        print(norm_factor)

        coord_df_noequil_sliced = coord_df_noequil[coord_df_noequil["E177"] > 0][["Frame","TrajNum","Z","ResidID"]]
        dunk_labels_ionlink2 = TestProject.aux_ts["E177-Ion Link"].join(dunking_labels(TestProject.dihe_ts["E177 Chi1-Chi2 Dihedral Angles"],
                           ["S1-Chi2", 'S2-Chi2', 'S3-Chi2', 'S4-Chi2']))
        super_coord_data = coord_df_noequil_sliced.merge(dunk_labels_ionlink2, on=["Frame","TrajNum"])
        super_coord_data["DunkCoord"] = super_coord_data.apply(coord_glustate, axis=1)

        f1, ax = plt.subplots()
        plot_mode_histogram(ax, super_coord_data, xlim=[-10,10], ylim=[0,1.0],
                             mode_colname = "DunkCoord",
                             all_binding_modes = [0,1,2],
                             x_label="position (Ang)",
                             y_label="probability (arb. units)",
                             norm_factor=norm_factor, sumdist_color=coord_df_color)

        f1.set_size_inches(18.5, 5.5)
        if len(series_to_loop_over) == 1:
            f1.savefig(TestProject.output_name+'_zhistogram_for_gludunklabels.pdf', dpi=200)
        else:
            if series_id == 0:
                f1.savefig(TestProject.output_name+'_SOD_zhistogram_for_gludunklabels.pdf', dpi=200)
            else:
                f1.savefig(TestProject.output_name+'_POT_zhistogram_for_gludunklabels.pdf', dpi=200)

