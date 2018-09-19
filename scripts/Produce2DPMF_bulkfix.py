""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates and
histograms of ion positions with state dividers plotted.

# Example script usage using Project configuration file:
python ${script} -c test2.cfg -ts "1st Shell K+ SF Coordination Extended"
python ${script} -c test3.cfg -ts "1st Shell Na+ SF Coordination Extended" "1st Shell K+ SF Coordination Extended"
"""

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from PandasMD.Initialize import Project
from itertools import product
from collections import defaultdict
from PandasMD.Coordination import mean_2d_axial_pmf
from PandasMD.Plot import plot_2d_pmfs
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
    help='a list of configfiles describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', nargs="+", type=str, required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    '-fname', dest='file_name', type=str, required=False,
    help='file name prefix for output')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path, coord_ts=args.series_name)
    print("Successfully Loaded Project: %s" % TestProject.name)

    all_series = []
    for series_id, series in enumerate(args.series_name):

        coord_df_noequil = prune_time(TestProject.coord_ts[series],
                                      TestProject.start_time,
                                      TestProject.end_time)
        coord_df2_trim = prune_column(coord_df_noequil, "Z", -10, 10) 
        all_series.append(coord_df2_trim)

    superdf = pd.concat(all_series)
    pmf = mean_2d_axial_pmf(superdf, histrange=[-10,10], histbins=100,)

    f1 = plt.figure()
    species_list=superdf["RowID"].unique()
    for p, (a,b) in enumerate(list(product(species_list, repeat=2))):
        # Work with a 2x2 grid always, keeps the size consistent
        # If we goto a larger number of ions, this is gotta be changed
        ax = f1.add_subplot(2,2,p+1)
        cax=plot_2d_pmfs(ax, pmf[(a,b)].transpose(),
                         hist_range=[-10,10,-10,10],
                         x_label="Outer ion "+b,
                         y_label="Inner ion "+a,
                         x_ticks=[2.0,4.0],
                         y_ticks=[2.0,4.0], vrange=[0,5])

    cbar_ax = f1.add_axes([0.85, 0.15, 0.05, 0.7])
    f1.colorbar(cax, cax=cbar_ax, ticks=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    cbar_ax.set_ylabel('free energy (kcal/mol)', rotation=270)

    f1.set_size_inches(18.5, 5.5)
    if args.file_name != None:
        f1.savefig(args.file_name+'_2d_zpmf_bulk_tiltfix.pdf', dpi=200)
    else:
        f1.savefig(TestProject.output_name+'_2d_zpmf_bulk_tiltfix.pdf', dpi=200)
