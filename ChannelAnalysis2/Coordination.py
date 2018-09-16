""" This script contains data-processing functionality for coordination
timeseries data. It produces numerous data series from the raw data. """

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from operator import gt
from collections import defaultdict
from ChannelAnalysis2.Initialize import Project
from itertools import combinations, product

def mean_axial_pmf(coord_df, axis_column=["Z"], histrange=[-10, 14],
                   histbins=300, min_binval=2, left_shift=True):
    """
    Computing a mean PMF is dangerous business. If you have
    undersampled regions of your reaction coordinate, you risk
    really fouling things up by taking the log. In this function
    we normalize by the number of samples in each trajectory,
    and the mean/sem is computed across all trajectories. The
    mean is also shifted so the average of the left most 5% bins
    is at zero. This rests on the assumption that you have equal
    sampling opportunity at this edge of the PMF (when 
    comparing PMFs of multiple datasets).

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    axis_column : list
        Column name in coord_df to histogram and compute PMF from.
    histrange : list
        Min and max values of PMF for the data in axis_column.
    histbins : int
        Number of histogram bins, determines jaggedness/smoothness.
    min_binval : int
        Histogram bins are not counted in the PMF for values less.
    left_shift : bool
        Shifts the y-val to zero based on 0-edge points, otherwise min-value
    """
    axial_histogram = {}
    for traj, data in coord_df.groupby(["TrajNum"]):
        histo, edges = np.histogram(data[axis_column],
                                    range=[histrange[0], histrange[1]],
                                    bins=histbins, normed=False)
        axial_histogram["Traj"+str(traj)] = histo

    histogram_df = pd.DataFrame(axial_histogram)
    histogram_df[histogram_df <= min_binval] = np.NaN
    histogram_df /= histogram_df.sum(axis=0)
    # -0.596 is R for kj/mol
    pmf_df = -0.596*np.log(histogram_df)
    pmf_df_mean = pmf_df.mean(axis=1)

    # Shift the PMF by the first 300*0.05 points
    # We want the energy to be zero'd on the left side.
    if left_shift:
        y_shift = pmf_df_mean[0:int(histbins*0.05)].mean()
    else:
        y_shift = pmf_df_mean.min()

    merged_df = pd.DataFrame(zip(pd.Series(edges[1:]),
                                 pmf_df_mean-y_shift,
                                 pmf_df.sem(axis=1)),
                             columns=["Bin","PMF","SEM"])

    return merged_df

def mean_2d_axial_pmf(coord_df, axis_column=["Z"], histrange=[-10, 10],
                   histbins=100, min_binval=2):
    """
    Takes all "inner ion" and "outer ion" pairs, taking into
    consideration their species type, and constructs a mean
    PMF of all possible pairings. Useful for examining
    ion conduction mechanisms. Note that if you have a mixed
    cation simulation, you need to pd.concat those data together.
    Note that here we did not calculate mean over trajectories,
    we just lumped all the data together.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    axis_column : list
        Column name in coord_df to histogram and compute PMF from.
    histrange : list
        Min and max values of PMF for the data in axis_column.
    histbins : int
        Number of histogram bins, determines jaggedness/smoothness.
    min_binval : int
        Histogram bins are not counted in the PMF for values less.
    """
    species_list=coord_df["RowID"].unique()

    # We need "Order/RowID" columns if we don't have them
    if 'Order' not in coord_df.columns:
        coord_df = macrostate_labels(coord_df)

    # Drop irrelevant columns, and sort by Order
    zz=coord_df[["TrajNum", "Frame",
                 "Z","Order",
                 "RowID"]].sort_values(by=["TrajNum",
                                           "Frame","Order"])

    # At each frame, extract Z coordinates and species name
    # and pair them up, saving Z values in the zpairs dict
    zhists = defaultdict(list)
    for traj, traj_group in zz.groupby(["TrajNum"]):
        #print(traj)
        zpairs = defaultdict(list)
        for name, group in traj_group.groupby(["Frame"]):
            order_pairs = combinations(range(len(group)), 2)
            zvals = group["Z"].values
            rowids = group["RowID"].values
            for pair in order_pairs:
                pair_str = str(rowids[pair[0]])+"-"+str(rowids[pair[1]])
                zpairs[pair_str+"-inner"].append(zvals[pair[0]])
                zpairs[pair_str+"-outer"].append(zvals[pair[1]])

        #print("Histogramming...")
        for p, (a,b) in enumerate(list(product(species_list, repeat=2))):
            if a != b:
                inner=zpairs[a+"-"+b+"-inner"]+zpairs[b+"-"+a+"-outer"]
                outer=zpairs[a+"-"+b+"-outer"]+zpairs[b+"-"+a+"-inner"]
            else:
                inner=zpairs[a+"-"+b+"-inner"]
                outer=zpairs[a+"-"+b+"-outer"]
            histo, xs, ys = np.histogram2d(inner,
                                           outer,
                                           range=[histrange, histrange],
                                           bins=[histbins,histbins],
                                           normed=False)
            zhists[(a,b)].append(histo)

    axial_pmfs = {}
    temp_dfs = {}
    min_val = 100
    for p, (a,b) in enumerate(list(product(species_list, repeat=2))):
        pmf_mean = -0.596*np.log(pd.Panel(zhists[(a,b)]).mean(axis=0).values)
        min_val = min(min_val, np.nanmin(pmf_mean))
        df=pd.DataFrame(pmf_mean, columns=ys[1:], index=xs[1:])
        temp_dfs[(a,b)]=df

    # This way we just ensure a single minimum across the set, instead
    # of them all being normalized separately! This has no consequences
    # for single ion species simulations.
    for p, (a,b) in enumerate(list(product(species_list, repeat=2))):
        axial_pmfs[(a,b)]=temp_dfs[(a,b)]-min_val

    return axial_pmfs

def mean_2d_pmf(coord_df, axis_columns=[["S1-Chi1"],["S1-Chi2"]],
                histrange=[0, 360],
                histbins=180, min_binval=2):
    """
    Same as mean_axial_pmf except in 2D! Useful for 2D rotamer
    heatmaps (Ramachandran). For some reason axis ranges are
    always symmetric in my studies, but it's easy to modify that.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    axis_columns : list
        Lists of lists, where each outer list contains the colnames
        for which we will unstack and histogram
    histrange : list
        Min and max values of PMF for the data in axis_column.
    histbins : int
        Number of histogram bins, determines jaggedness/smoothness.
    min_binval : int
        Histogram bins are not counted in the PMF for values less.
    """

    histogram_2d = {}
    for traj, data in coord_df.groupby(["TrajNum"]):
        for subunit, (xcol, ycol) in enumerate(zip(*axis_columns)):
        # Note that x and y must be the same length, so two at once
            xvals = data[xcol]
            yvals = data[ycol]
            #xvals = data[axis_columns[0]].unstack()
            #yvals = data[axis_columns[1]].unstack()

            # Things get weird if there is no data. We're not
            # doing SEM, so we don't actually care how many
            # datasets are added.
            if len(xvals) > 10:
                histo, xs, ys = np.histogram2d(xvals, yvals,
                                               range=[histrange,
                                                      histrange],
                                               bins=[histbins,histbins],
                                               normed=True)
                #histo[histo <= min_binval] = np.NaN
                #histo /= np.nansum(histo,axis=(0,1))
                histogram_2d["Traj"+str(traj)+"_"+str(subunit)] = histo

    #pmf_panel = -0.596*np.log(pd.Panel(histogram_2d))
    #pmf_mean = pmf_panel.mean(axis=0).values
    pmf_mean = -0.596*np.log(pd.Panel(histogram_2d).mean(axis=0).values)
    min_val = np.nanmin(pmf_mean)
    return pd.DataFrame(pmf_mean-min_val, columns=ys[1:], index=xs[1:])

def all_occupancy_ts(coord_df, nonzero_columns=None):
    """
    An occupancy timeseries produces a count of objects over time
    that were tracked in a geometric region of simulation. In ion channel
    simulations, this is typically the number of ions in the central pore.
    With the nonzero_columns argument, a subset of the total occupancy
    can be computed wherein a object is only counted if any column in
    its coordination list is non-zero. Returns a dataframe of all timeseries.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    nonzero_columns : list
        Column names where values must be non-zero to count as occupancy.

    """
    # If non-zero columns are specified, extract row subset.
    # TODO: This omits rows without coordination, actually wrong!
    if nonzero_columns != None:
        # Using the strategy suggested here:
        # http://stackoverflow.com/questions/13611065/
        nonzero_masks = [gt(coord_df[col],0) for col in nonzero_columns]
        coord_df = coord_df[(np.logical_or(*nonzero_masks))]

    # Multiple ions can exist at a given Traj and Time key, do a count
    # of these occurrences with a given Traj-Time key, and make a new dataframe
    occ_gb = coord_df.groupby(["TrajNum","Time"]).size()
    occ_df = pd.DataFrame(occ_gb, columns = ['Occ']).reset_index()

    # A pivot is necessary for a datatype with traj columns and rows
    # as occupancy vs. time. This makes it trivial to average across that axis
    return occ_df.pivot(index='Time', columns= 'TrajNum', values="Occ")


def mean_occupancy_ts(coord_df, nonzero_columns=None):
    """
    Produces a timeseries of mean/std/sem statistics on a coordination
    dataframe. Returns mean, std.dev, and sem of the values of interest.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    nonzero_columns : list
        Column names where values must be non-zero to count as occupancy.

    """

    occ_df_pivot = all_occupancy_ts(coord_df, nonzero_columns)

    # Build a new statistics dataframe (with NaNs)
    occ_df_stats = pd.concat([
              occ_df_pivot.mean(axis=1),
              occ_df_pivot.std(axis=1),
              occ_df_pivot.sem(axis=1)],
              axis=1)
    occ_df_stats.columns = ["Mean","STDEV","SEM"]

    return occ_df_stats

def dunking_labels(dihe_df, dihe_colnames, state_divider=[240,],
                            dihe_colnames2=None, state_divider2=[250,]):
    """
    Returns a state label for a dihedral angle based on
    state boundaries. By convention, lower angles result in high
    state indices. We used to do 180, but now 240 is the new hotness.

    Parameters
    ----------
    dihe_df : DataFrame
        This is the master coordination dataframe of the project.
    dihe_colnames : list
        A list of columns that contain dihedral angles.

    """
    # We reverse the states list because of convention, < 180
    # means "dunked", which we like to refer to as a higher number
    states = list(reversed([0]+state_divider+[360]))
    states2 = list(reversed([0]+state_divider2+[360]))

    # Subtract 1 just so we're 0-bin indexed
    dbool = pd.DataFrame(np.digitize(dihe_df[dihe_colnames], states), index=dihe_df.index)-1
    # The old "binary" way
    #dbool = (dihe_df[dihe_colnames] < state_divider).astype(int)

    # Sometimes we have additional dunking criteria, specified by the dihe_colnames2
    # table. For instance, Chi1 as well as Chi2.
    if dihe_colnames2 != None:
        # You need to look at the Ramachandran map to believe this,
        # here row[0] = chi1, row[1] = chi2, and 1 means "DUNK" (less than cutoff)
        def decide_dunking(row):
            if row[0] == 1 and row[1] == 0:
                return 0
            else:
                return 1

        # Extract and merge the chi1 columns with chi2
        dbool2 = pd.DataFrame(np.digitize(dihe_df[dihe_colnames2], states2), index=dihe_df.index)-1
        db=dbool.join(dbool2, lsuffix="_chi2", rsuffix="_chi1")

        for num in range(len(dbool.columns)):
            p = [str(num)+"_chi1", str(num)+"_chi2"]
            dbool[num] = db[p].apply(lambda row: decide_dunking(row), axis=1)

    # ALL only makes sense for binary dunking states, sorry!
    dbool["ALL"] = dbool.sum(axis=1)
    dbool.columns = ["S"+str(num)+"-dunked" for num in range(1,1+len(dihe_colnames))]+["SALL-dunked"]
    return dbool

    def coord_bool(row):
        z=row["Z"]
        slbl=row[coord_col]

        if ((slbl == "0011") or
            (slbl == "0111") or
            (slbl == "0001")):
            return "LT"
            print(row, slbl)
            return "NONE"

    return coord_df.apply(lambda row: coord_bool(row), axis=1)

def macrostate_occupancy_ts(coord_df, occ=2, prime=True, colnames=None):
    """
    Produces a dataframe with only axial Z values for ions
    at a particular macrostate (defined by occupancy and prime variables)

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    occ : int
        This holds the channel occupancy variable in "Occupancy" column
    prime : bool
        This holds the channel prime state (if ion is in central cavity)
    colnames : list
        This holds the column names for all ions at an occupancy state

    """

    if colnames==None:
        colnames=["Red","Green","Blue","Purple","Orange"][:occ] 

    # We need "Order/RowID" columns if we don't have them
    if 'Order' not in coord_df.columns:
        coord_df = macrostate_labels(coord_df)

    # This block unstacks the Z of red and green ions, for non-prime
    macrostate_bool = (coord_df["Occupancy"]==occ) & (coord_df["Prime"] == prime)
    coord_df=coord_df[macrostate_bool][["Z","Order","TrajNum",
                                        "Time"]].set_index(["Time",
                                                            "TrajNum","Order"]).unstack()
    coord_df.columns = coord_df.columns.droplevel(0)
    coord_df.columns = ["Red","Green","Blue","Purple","Orange"][:occ] 
    return coord_df.reset_index()

def coordination_labels(coord_df, coord_colnames):
    """
    Produces a series that uses boolean 0 or >0 labels
    for each of the coordination columns in a dataframe

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    coord_colnames : list
        A list of columns to run boolean row-wise testing on.

    """

    t = pd.concat([coord_df[c] > 0 for c in coord_colnames], axis=1)
    return t.applymap(lambda x: str(int(x))).apply(np.sum, axis=1)

def OPLS_mode_coordination_labels(coord_df, coord_col="CoordLabel"):
    """
    Uses a series of boolean 0 or >0 labels to classify
    ions positions into distinct binding modes (also using z).
    Very specific to Nav sodium channel SELT coordination.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    """

    def coord_bool(row):
        z=row["Z"]
        slbl=row[coord_col]

        if ((slbl == "0011") or
            (slbl == "0111") or
            (slbl == "0001")):
            return "LT"
        elif ((slbl == "0110") or
              (slbl == "0000" and z > -2 and z < 0) or
              (slbl == "0010") or
              (slbl == "1010") or
              (slbl == "0101") or
              (slbl == "1110")):
            return "EL"
        elif ((slbl == "1000") or
              (slbl == "1100")):
            return "S"
        elif ((slbl == "0100") or
              (slbl == "0000" and z > -5 and z <= -2)):
            return "E"
        elif ((slbl == "0000" and z >= 0)):
            return "CC"
        elif (slbl == "0000" and z <= -5):
            return "EC"
        else:
            print(row, slbl)
            return "NONE"

    return coord_df.apply(lambda row: coord_bool(row), axis=1)
    

def mode_coordination_labels(coord_df, coord_col="CoordLabel"):
    """
    Uses a series of boolean 0 or >0 labels to classify
    ions positions into distinct binding modes (also using z).
    Very specific to Nav sodium channel SELT coordination.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    """

    def coord_bool(row):
        z=row["Z"]
        slbl=row[coord_col]
        slbl2=row[coord_col+"_2nd"]
        if ((slbl == "0000" and slbl2 == "0011" and z > 1) or
            (slbl == "0000" and slbl2 == "0111" and z > 1) or
            (slbl == "0000" and slbl2 == "0010" and z > 1) or
            (slbl == "0000" and slbl2 == "0100" and z > 1) or
            (slbl == "0000" and slbl2 == "0101" and z > 1) or
            (slbl == "0001") or
            (slbl == "0011")):
            return "LT"
        #elif ((slbl == "0110") or
        #      (slbl == "0010") or
        #      (slbl == "0111") or
        #      (slbl == "0101") or
        #      (slbl == "0000" and slbl2 == "1111") or
        #      (slbl == "0000" and slbl2 == "0110") or
        #      (slbl == "0000" and slbl2 == "1110") or
        #      (slbl == "0000" and slbl2 == "1010") or
        #      (slbl == "0000" and slbl2 == "1100") or
        #      (slbl == "0000" and slbl2 == "1000") or
        #      (slbl == "0000" and slbl2 == "1101") or
        #      (slbl == "0000" and slbl2 == "1011") or
        #      (slbl == "0100" and z > 0)):
        elif ((slbl == "0110") or
              (slbl == "0111") or
              (slbl == "0101") or
              (slbl == "0000" and slbl2 == "1111") or
              (slbl == "0000" and slbl2 == "1110") or
              (slbl == "0000" and slbl2 == "1010") or
              (slbl == "0000" and slbl2 == "1100") or
              (slbl == "0000" and slbl2 == "1000") or
              (slbl == "0000" and slbl2 == "1101") or
              #(slbl == "0000" and slbl2 == "1011")):
              (slbl == "0000" and slbl2 == "1011") or
              (slbl == "0100" and z > 1)):
            return "EL"
        # Lump S-only and L-only into E, and EL respectively
        elif ((slbl == "0010") or
              (slbl == "0000" and slbl2 == "0110")):
            #return "L"
            return "EL"
        elif slbl == "1000":
            #return "EC"
            return "S"
            #return "E"
        elif ((slbl == "1100") or
              (slbl == "1110") or
              (slbl == "1111") or
              (slbl == "1101") or
              (slbl == "1010") or
              (slbl == "1011") or
              (slbl == "0100" and z <= 1) or
              #(slbl == "0100") or
              (slbl =="0000" and -5 <= z <= 1)):
            return "E"
        elif ((slbl == "0000" and slbl2 == "0001") or
              (slbl == "0000" and slbl2 == "0100" and z > 5) or
              (slbl == "0000" and slbl2 == "0100" and z > 5) or
              (slbl == "0000" and slbl2 == "0000" and z > 0)):
            return "CC"
        elif (slbl == "0000" and z < -5):
            return "EC"
        else:
            print(row, slbl, slbl2)
            return "NONE"

    return coord_df.apply(lambda row: coord_bool(row), axis=1)

def occupancy_populations_pertraj(coord_df, coord_col="ModeLabel",
                                  coord_colvals=['EC','S', 'E', 'EL','LT','CC'],
                                  coord_sfcolvals=['S','E','EL','LT']):
    """
    Produces a dictionary of coord_colval keys with dataframes representing
    ion occupancy

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    coord_col : str
        Column name in coord_df to compute ion occupancy using.
    coord_colvals : list
        List of values which may or may not exist as coord_col values.

    """
    tgroup = coord_df.groupby(["TrajNum",
                               "Time",coord_col]).size().unstack().fillna(0)[coord_colvals]
    tgroup["CHAN"]=tgroup.sum(axis=1)
    tgroup["SF"]=tgroup[coord_sfcolvals].sum(axis=1)

    if coord_colvals == None:
        coord_colvals = tgroup.columns

    return _occupancy_populations_normalize(tgroup, coord_colvals=["CHAN","SF"]+coord_colvals)

def _occupancy_populations_normalize(mode_df,
                                     coord_colvals=['EC','S', 'E', 'EL','LT','CC']):
    """
    This is a helper method for computing occupancy populations
    once you've converted a normal coordination dataframe
    to a mode dataframe of type:

    ModeLabel        CHAN  SF  EC  S  E  EL  LT  CC
    TrajNum Time
    1       150.00      2   2   0  0  1   1   0   0
            150.02      2   2   0  0  1   1   0   0
            150.04      2   2   0  0  1   1   0   0
            ...

    Parameters
    ----------
    mode_df : DataFrame
        This is the mode dataframe, produced using occupancy_populations_pertraj
    coord_col : str
        Column name in coord_df to compute ion occupancy using.
    coord_colvals : list
        List of values which may or may not exist as coord_col values.

    """
    modes_per_traj=defaultdict(dict)

    for traj, data in mode_df.reset_index().groupby(["TrajNum"]):
        for colname in coord_colvals:
            if colname in data:
                z=data[colname].value_counts()
                modes_per_traj[colname][traj]=z/z.sum()
            else:
                modes_per_traj[colname][traj]=pd.Series([1.0], index=[0])

    return_df = {}
    for colname, trajdata in modes_per_traj.iteritems():
        return_df[colname] = pd.DataFrame(trajdata).fillna(0)
        return_df[colname].columns.name = "TrajNum"
        return_df[colname].index.name = "Occ"

    return return_df

def joint_occupancy_populations_pertraj(coord_dfs, coord_col="ModeLabel",
                                  coord_colvals=['EC','S', 'E', 'EL','LT','CC'],
                                  coord_sfcolvals=['S','E','EL','LT']):
    """
    Produces a dictionary of coord_colval keys with dataframes representing
    joint ion occupancy

    Parameters
    ----------
    coord_dfs : list of DataFrame
        List of coordination dataframe of the project for each ion type
    coord_col : str
        Column name in coord_df to compute ion occupancy using.
    coord_colvals : list
        List of values which may or may not exist as coord_col values.

    """
    num_species = len(coord_dfs)
    frac_species = 1.0/num_species

    if coord_colvals == None:
        coord_colvals = tgroup.columns

    # The following unstacking procedure gives us two dataframes
    # with the type,
    #
    # ModeLabel        CHAN  SF  EC  S  E  EL  LT  CC
    # TrajNum Time                                   
    # 1       150.00      2   2   0  0  1   1   0   0
    #         150.02      2   2   0  0  1   1   0   0
    #  ...
    all_unstacked = []
    for type_id, coord_df in enumerate(coord_dfs):
        tgroup = coord_df.groupby(["TrajNum","Time",coord_col]).size().unstack().fillna(0)
        tgroup["CHAN"]=tgroup.sum(axis=1)
        tgroup["SF"]=tgroup[coord_sfcolvals].sum(axis=1)
        all_unstacked.append(tgroup)

    # Construct a panel of dataframes and sum across them
    # To get total occupancy counts at all traj/time pairs
    # Note: this can take a while...
    sum_unstacked = pd.Panel({n:a for n,a in enumerate(all_unstacked)}).sum(axis=0)

    # Now we'll compute the normalized occupancy of each site which
    # returns data like this for each column, where 3 4 6 are trajectory IDs
    #TrajNum       3         4      6  
    #Occ                               
    #0        0.995733  0.998867  0.967
    #1        0.004267  0.001133  0.033
    #2        0.000000  0.000000  0.000
    total_occ_dict = _occupancy_populations_normalize(sum_unstacked,
                                                   coord_colvals=["CHAN","SF"]+coord_colvals)

    # We only get NaNs when the sum of a site is zero. This is a special
    # case that we can decide how to treat. The best approach is to say that
    # occupancy is split between ion types (0.5, 0.5) in this case,
    fractional_occupancy = [df.div(sum_unstacked,
                                   fill_value=0).fillna(frac_species) for df in all_unstacked]
    # TODO: Make it so the Type is the same as the input data column
    for t,df in enumerate(fractional_occupancy):
        df["Type"]=t

    merged_frac_occ = pd.concat(fractional_occupancy)
    merged_frac_occ_w_sum = merged_frac_occ.join(sum_unstacked,
                                                 rsuffix="_SUM", how="right")

    return_df = {}
    for col in ["CHAN","SF"]+coord_colvals:
        cc = merged_frac_occ_w_sum.reset_index()[["TrajNum",col+"_SUM","Type",col]]
        cc.columns=["TrajNum", "Occ", "Type", "Percent"]

        # By using a PivotTable, we can summarize our data with TrajNum across columns
        # which ends up being the same shape as total_occ_dict DFs. Notice that this process
        # can introduce missing values. Return this multiplied by actual occupancy percent!
        #
        #            Percent                    
        # TrajNum        3         4         6  
        # Occ Type                              
        # 0   0     0.500000  0.500000  0.500000
        #     1     0.500000  0.500000  0.500000
        # 1   0     0.359375  0.058824  0.783838
        #     1     0.640625  0.941176  0.216162
        # 2   0          NaN       NaN       NaN
        #     1          NaN       NaN       NaN
        col_pivot = pd.pivot_table(cc, columns="TrajNum",
                                   index=["Occ","Type"]).fillna(frac_species)

        # Multiplying total_occ_dict results in a broadcast to the two atom types
        return_df[col] = col_pivot*total_occ_dict[col]

    return return_df

def species_macrostate_labels(coord_df, drop_prime=False):
    """
    Produces two columns called Order and Prime that are assigned to each row
    that describe the number of ions at that step, and prime if there is an
    ion in the lowest order state. This differs from macrostate_labels
    because we consider the ion species in the RowID column.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    drop_prime : bool
        Enabling drop_prime simply sets all prime values to false.
    """
    # TODO: Use groupby()["col"].rank() here, except it's slow as heck for >1M dfs?
    gdf = coord_df.sort_values(by=["TrajNum","Time","Z"], ascending=(True,True,False))

    sizes = gdf.groupby(["TrajNum","Time"]).size().values
    gdf["Order"] = np.arange(sizes.sum()) - np.repeat(sizes.cumsum() - sizes, sizes)
    size_df = gdf.groupby(["TrajNum","Time"]).size().reset_index(name="Occupancy")

    gdf_size = pd.merge(gdf, size_df, on=["TrajNum","Time"])

    # Comment out this conditional if you don't care about prime or unprime ions
    gdf_lowestorder = gdf_size[gdf_size["Order"] == 0][["Time","TrajNum","S178",
                                                        "E177","E177p","L176","T175",
                                                        "Occupancy","Z"]]
    gdf_highestorder = gdf_size.groupby(["TrajNum",
                                         "Time"]).last().reset_index()[["Time","TrajNum","S178",
                                                                        "E177","E177p","L176","T175",
                                                                        "Occupancy","Z","RowID"]]
    gdf_highestorder["OuterRowID"] = gdf_highestorder["RowID"]

    if drop_prime == True:
        gdf_lowestorder["Prime"] = False
    else:
        # Is an inner ion actually bound in second shell to THR (with less than 4 coordination)?
        gdf_lowestorder["Prime"] = pd.Series((gdf_lowestorder["T175"] <= 4) &
                                             (gdf_lowestorder["L176"] == 0) &
                                             (gdf_lowestorder["E177p"] == 0) &
                                             (gdf_lowestorder["E177"] == 0) &
                                             (gdf_lowestorder["S178"] == 0) &
                                             (gdf_lowestorder["Z"] > 0.5))

    gdf_lowestorder = pd.merge(gdf_lowestorder,
                               gdf_highestorder[["Time","TrajNum",
                                                 "OuterRowID"]], on=["TrajNum","Time"])

    def label_macrostate(row):
        if row["Prime"]:
            return str(row["OuterRowID"])+"_"+str(row["Occupancy"]-1)+"'"
        else:
            return str(row["OuterRowID"])+"_"+str(row["Occupancy"])

    gdf_lowestorder["OccMacrostate"] = gdf_lowestorder.apply(label_macrostate, axis=1)

    return pd.merge(gdf, gdf_lowestorder[["TrajNum","Time","Prime",
                                          "OuterRowID","Occupancy","OccMacrostate"]],
                    on=["TrajNum","Time"])

def macrostate_labels(coord_df, drop_prime=False):
    """
    Produces two columns called Order and Prime that are assigned to each row
    that describe the number of ions at that step, and prime if there is an
    ion in the lowest order state

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    drop_prime : bool
        Enabling drop_prime simply sets all prime values to false.
    """
    # TODO: Use groupby()["col"].rank() here, except it's slow as heck for >1M dfs?
    gdf = coord_df.sort_values(by=["TrajNum","Time","Z"], ascending=(True,True,False))

    sizes = gdf.groupby(["TrajNum","Time"]).size().values
    gdf["Order"] = np.arange(sizes.sum()) - np.repeat(sizes.cumsum() - sizes, sizes)
    size_df = gdf.groupby(["TrajNum","Time"]).size().reset_index(name="Occupancy")

    gdf_size = pd.merge(gdf, size_df, on=["TrajNum","Time"])

    # Comment out this conditional if you don't care about prime or unprime ions
    gdf_lowestorder = gdf_size[gdf_size["Order"] == 0][["Time","TrajNum","S178",
                                                        "E177","E177p","L176","T175",
                                                        "Occupancy","Z"]]
    gdf_highestorder = gdf_size.groupby(["TrajNum",
                                         "Time"]).last().reset_index()[["Time","TrajNum","S178",
                                                                        "E177","E177p","L176","T175",
                                                                        "Occupancy","Z"]]

    if drop_prime == True:
        #gdf_highestorder["OuterPrime"] = False
        # Is an outer ion actually bound in second shell to SER (with less than 4 coordination)?
        gdf_highestorder["OuterPrime"] = pd.Series((gdf_highestorder["T175"] == 0) &
                                             (gdf_highestorder["L176"] == 0) &
                                             (gdf_highestorder["E177p"] == 0) &
                                             (gdf_highestorder["E177"] <= 2) &
                                             #(gdf_highestorder["E177"] == 0) &
                                             (gdf_highestorder["S178"] <= 4) &
                                             (gdf_highestorder["Z"] < -5))
        gdf_lowestorder["Prime"] = False
    else:
        # Is an outer ion actually bound in second shell to SER (with less than 4 coordination)?
        gdf_highestorder["OuterPrime"] = pd.Series((gdf_highestorder["T175"] == 0) &
                                             (gdf_highestorder["L176"] == 0) &
                                             (gdf_highestorder["E177p"] == 0) &
                                             (gdf_highestorder["E177"] <= 2) &
                                             #(gdf_highestorder["E177"] == 0) &
                                             (gdf_highestorder["S178"] <= 4) &
                                             (gdf_highestorder["Z"] < -5))

        # Is an inner ion actually bound in second shell to THR (with less than 4 coordination)?
        gdf_lowestorder["Prime"] = pd.Series((gdf_lowestorder["T175"] <= 4) &
                                             (gdf_lowestorder["L176"] == 0) &
                                             (gdf_lowestorder["E177p"] == 0) &
                                             (gdf_lowestorder["E177"] == 0) &
                                             (gdf_lowestorder["S178"] == 0) &
                                             #(gdf_lowestorder["Z"] > 0.5))
                                             (gdf_lowestorder["Z"] > 5))

    gdf_lowestorder = pd.merge(gdf_lowestorder,
                               gdf_highestorder[["Time","TrajNum",
                                                 "OuterPrime"]], on=["TrajNum","Time"])

    # These two functions denote the "classic" 2nd shell designation of macrostate
    # or a new one that checks the outer ion.

    #def label_macrostate(row):
    #    if row["Prime"]:
    #        return str(row["Occupancy"]-1)+"'"
    #    else:
    #        return str(row["Occupancy"])
    def label_macrostate(row):
        if row["Prime"] and row["OuterPrime"]:
            #print("Impossible, ", row)
            return str(row["Occupancy"]-2)+"'"
        elif row["Prime"]:
            return str(row["Occupancy"]-1)+"'"
        elif row["OuterPrime"]:
            #print("OP Impossible, ", row)
            return str(row["Occupancy"]-1)
        else:
            return str(row["Occupancy"])
    gdf_lowestorder["OccMacrostate"] = gdf_lowestorder.apply(label_macrostate, axis=1)

    return pd.merge(gdf, gdf_lowestorder[["TrajNum","Time","Prime",
                                          "OuterPrime","Occupancy","OccMacrostate"]],
                    on=["TrajNum","Time"])

def ordering_labels(coord_df,col="ResidID"):
    """
    Produces an ordering column, where a tuple of sorted values is stored for
    each frame. Useful if you wanted the sequence of residues, or sequence of Z.
    Warning: this is horribly slow.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    col : str
        The values of this col for entire group will be put into this tuple
    """
    gdf = coord_df.sort_values(by=["TrajNum","Frame","Z"], ascending=(True,True,False))
    lbls= []
    # We now TrajNum, Frame, Ordering pairs now
    for k,v in gdf.groupby(["TrajNum","Frame"]):
        lbls.append([k[0],k[1],tuple(v[col].values)]) 
    lbl_df = pd.DataFrame(lbls, columns=["TrajNum","Frame","Ordering"])

    return pd.merge(gdf, lbl_df, on=["TrajNum","Frame"])

if __name__ == '__main__':
    parser = ArgumentParser(
    description='This script extracts basic statistics of coordination \
    with attention to statistics across multiple timeseries.')
    parser.add_argument(
    '-c', dest='cfg_path', type=str, required=True,
    help='a configfile describing data paths and parameters for a project')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)
    #print(TestProject.coord_ts)

    #print(mean_occupancy_timeseries(TestProject.coord_ts.values()[0]))
    occ_ts_data = mean_occupancy_ts(TestProject.coord_ts["1st Shell Na+ SF Coordination"],nonzero_columns=["E177","L176"])
    print(occ_ts_data)
    print(occ_ts_data.head())
    print(occ_ts_data.tail())
    occ_ts_data.plot()
