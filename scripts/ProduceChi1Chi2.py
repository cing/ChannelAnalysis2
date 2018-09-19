""" This script contains data-processing functionality for making
Chi1 and Chi2 maps of E177 conformations within the SF, aswell as
computing populations along certain dividing lines.

# Example script usage using Project configuration file:
#python ${script} -c test2_jcdr.cfg  -ts "E177 Chi1-Chi2 Dihedral Angles"
#python ${script} -c test_NNNH.cfg  -ts "E177 Chi1-Chi2 Dihedral Angles" -div 120 240
"""

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from ChannelAnalysis2.Initialize import Project
from collections import defaultdict
from itertools import chain, repeat, islice, groupby
from ChannelAnalysis2.Coordination import dunking_labels, mean_2d_pmf
from ChannelAnalysis2.Plot import plot_2d_pmfs
import matplotlib.pyplot as plt

def prune_time(df, start, end):
    return df[(df["Time"] >= start) & (df["Time"] <= end)]

def prune_column(df, col, start, end):
    return df[(df[col] >= start) & (df[col] <= end)]

# This helper function will allow me to iterate over a fixed window
# http://stackoverflow.com/q/6998245/1086154
def window(seq, size=2, fill=0, fill_left=False, fill_right=False):
    """ Returns a sliding window (of width n) over data from the iterable:
      s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    ssize = size - 1
    it = chain(
      repeat(fill, ssize * fill_left),
      iter(seq),
      repeat(fill, ssize * fill_right))
    result = tuple(islice(it, size))
    if len(result) == size:  # `<=` if okay to return seq if len(seq) < size
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

if __name__ == '__main__':
    parser = ArgumentParser(
    description='This script extracts basic transition statistics of \
    coordination with attention to statistics across multiple timeseries.')
    parser.add_argument(
    '-c', dest='cfg_path', type=str, required=True,
    help='a configfile describing data paths and parameters for a project')
    parser.add_argument(
    '-ts', dest='series_name', type=str, required=True,
    help='column name of coordination data to compute rates')
    parser.add_argument(
    #'-div', dest='dividers', nargs="+", type=int, default=[180.,],
    '-div', dest='dividers', nargs="+", type=int, default=[240.,],
    help='splits in dihedral space to compute occupancy')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % (TestProject.name))

    f1, ax = plt.subplots(1)

    # This is a dictionary that holds the previous bar chart y values
    # This will get populated after series_id > 0
    yoffset={}
    ptitles={}

    dihe_df2_merge = prune_time(TestProject.dihe_ts[args.series_name],
                                TestProject.start_time,
                                TestProject.end_time)

    # No matter what, get the labels right, later we can do a subset
    chi1_cols = ["S1-Chi1", 'S2-Chi1', 'S3-Chi1', 'S4-Chi1']
    chi2_cols = ["S1-Chi2", 'S2-Chi2', 'S3-Chi2', 'S4-Chi2']

    dunk_labels = dunking_labels(dihe_df2_merge, chi2_cols,
                                 state_divider=args.dividers,#)
                                 dihe_colnames2=chi1_cols,)
    dihe_df2_merge = dihe_df2_merge.join(dunk_labels)

    #dunk_cols = ["S1-dunked"]
    #chi1_cols = ["S1-Chi1"]
    #chi2_cols = ["S1-Chi2"]
    #dunk_cols = ["S2-dunked","S3-dunked","S4-dunked"]
    #chi1_cols = ['S2-Chi1', 'S3-Chi1', 'S4-Chi1']
    #chi2_cols = ['S2-Chi2', 'S3-Chi2', 'S4-Chi2']
    dunk_cols = ["S1-dunked","S2-dunked","S3-dunked","S4-dunked"]
    chi1_cols = ["S1-Chi1", 'S2-Chi1', 'S3-Chi1', 'S4-Chi1']

    dunk_per_traj = {}
    for traj, data in dihe_df2_merge.groupby("TrajNum"):
        dunk_per_traj[traj] = pd.Series(pd.Categorical(data[dunk_cols].unstack(),
                                        categories=range(len(args.dividers)+1),
                                        ordered=False)).value_counts(normalize=True)

    label_means = pd.DataFrame(dunk_per_traj).mean(axis=1).to_dict()
    label_sems = pd.DataFrame(dunk_per_traj).sem(axis=1).to_dict()

    print("Mean dunking is: ", 4*label_means[1], 4*label_sems[1])
    # End dunking statistics calculation

    pmf = mean_2d_pmf(dihe_df2_merge, axis_columns=[chi1_cols, chi2_cols],
                      histrange=[0, 360], histbins=180, min_binval=2)

    cax=plot_2d_pmfs(ax, pmf,
                     hist_range=[0,360,0,360],
                     x_label="Chi1 (degrees)",
                     y_label="Chi2 (degrees)",
                     x_ticks=[30,60],
                     y_ticks=[30,60], vrange=[0,5])

    # This prints text on the 2D histogram that indictes the percentage
    # data in that dihedral space defined between the dividers
    num_regions = len(args.dividers)+1
    # This iterates over the state divider limits (top and bottom)
    # in order to position text within those boundaries.
    all_dividers = [0]+args.dividers+[360]
    
    for range_id, range_vals in enumerate(window(reversed(all_dividers))):
        y_position = ((range_vals[0]+range_vals[1])/2)/360.0
        try:
            range_avg = 100*label_means[range_id]
        except:
            range_avg = 0.0
        try:
            range_sem = 100*label_sems[range_id]
        except:
            range_sem = 0.0
            
        plt.text(0.90, y_position,
         r"{:.1f}% $\pm$ {:.1f}%".format(range_avg, range_sem),
         #weight="bold",
         ha='center', va='center', transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.5))

    cbar = f1.colorbar(cax, ticks=[0, 1, 2, 3, 4, 5, 6])
    f1.set_size_inches(11, 11)
    #f1.savefig(TestProject.output_name+'_chi1_chi2_heatmap_subBCD.pdf', dpi=200)
    #f1.savefig(TestProject.output_name+'_chi1_chi2_heatmap_subA.pdf', dpi=200)
    f1.savefig(TestProject.output_name+'_chi1_chi2_heatmap.pdf', dpi=200)
    #f1.savefig(TestProject.output_name+'_chi1_chi2_heatmap_triple.pdf', dpi=200)
    #f1.savefig(TestProject.output_name+'_chi1_chi2_heatmap_triple_subBCD.pdf', dpi=200)
    #f1.savefig(TestProject.output_name+'_chi1_chi2_heatmap_triple_subA.pdf', dpi=200)
