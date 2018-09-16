""" This script contains data-processing functionality for counting
transitions in coordination timeseries data. It produces rates. """
#TODO This function has major redundancy, needs a refactor

from __future__ import print_function, division, absolute_import
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from ChannelAnalysis2.Initialize import Project
import itertools
from collections import defaultdict

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

def mode_coordstate_transitions_per_trajres(coord_df,
                                  dwell_cut=2,
                                  states_per_transition=2,
                                  transition_column="ModeLabel",
                                  return_stats=True):
    """
    A state transition dictionary produces a count of transitions or
    transition rates for the solute defined in coord_df that was tracked
    in a geometric region of simulation as it traverses states defined
    by the dividers list. Note that left and right of the dividers still
    indicate states. In ion channel simulations, this is typically an
    ion traversing specific regions of the permeation pore in 1 dimension.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    dwell_cut : int
        States less than dwell_cut timesteps are pruned.
    states_per_transition : int
        Number of states that make a transition, two state is default.
    return_stats : bool
        True returns count of transitions, False returns transitions
    """

    # Drop nuisance columns.
    classified_df = coord_df.loc[:,("TrajNum","Frame","ResidID",transition_column)]

    # Iterate over each resid in each trajectory
    classified_groups = classified_df.groupby(["TrajNum","ResidID"])

    trans_counts_per_traj = pd.DataFrame()
    trans_per_traj = []

    for trajresid, trajresid_ts in classified_groups:
        # trajresid is a tuple of format (trajnum, resid)
        #print(trajresid),

        # Tiny bit of magic here, groupby consecutive time chunks:
        # http://stackoverflow.com/questions/26121668/slice-pandas-dataframe-in-groups-of-consecutive-values
        contig_timeblocks = trajresid_ts['Frame'] - np.arange(trajresid_ts.shape[0])

        for k, g in trajresid_ts.groupby(contig_timeblocks):
            trans_count = defaultdict(int)

            # Drop outside of pandas and use itertools to groupby zrange ID
            # until pandas supports this natively (Pandas 0.16 maybe?)
            grouped_iter = itertools.groupby(g[[transition_column,"Frame"]].values, key=lambda col: col[0])

            # Lose the iterator, now we have a list of format (zrangeid, (zrangeid, time))
            grouped_zranges = [(zid, list(dwell)) for zid, dwell in grouped_iter]

            # Preserve the first time label, sorry the indexing is so bad!
            # Now we have a list of format (zrangeid, start_time, dwell_time)
            condensed_zranges = [(zpair[0], zpair[1][0][1], len(zpair[1]))
                                 for zpair in grouped_zranges]

            # Remove short-lived states with a quick pruning.
            cutoff_zranges = [zpair for zpair in condensed_zranges if zpair[2] >= dwell_cut]

            # Now we stitch together pairs with identical zrange ID by
            # iteratively merging time blocks and adjusting the dwell time
            # of the first

            #print("CONDENSED: ", cutoff_zranges)
            joined_zranges = []
            if len(cutoff_zranges) > 1:
                # Iteratively pop off a contiguous timeblock from the list
                # and merge it with the next timeblock if it shares a common
                # ZRange
                curr_zrange = cutoff_zranges.pop(0)
                while len(cutoff_zranges) > 0:
                    next_zrange = cutoff_zranges.pop(0)
                    if curr_zrange[0] == next_zrange[0]:
                        curr_zrange = (curr_zrange[0], curr_zrange[1],
                                       sum(next_zrange[1:3])-curr_zrange[1])
                    else:
                        joined_zranges.append(curr_zrange)
                        curr_zrange = next_zrange
                joined_zranges.append(curr_zrange)

            #print("JOINED: ",joined_zranges)
            # If you want stats, just increment a counter
            # If you want transitions, take the whole before-after pair
            if return_stats:
                for states in window(joined_zranges, states_per_transition):
                    # Here state[0] is the state label
                    state_trans_str = "-".join([str(state[0]) for state in states])
                    trans_count[state_trans_str] += 1

                #for before, after in window(joined_zranges, 2):
                #    trans_count[str(before[0])+"-"+str(after[0])]+=1
            else:
                for states in window(joined_zranges, states_per_transition):
                    # Just make sure we have all unique states in this set,
                    # otherwise, it's just a recrossing!
                    if len(set([state[0] for state in states])) == states_per_transition:
                        trans_per_traj.append(list(itertools.chain.from_iterable(states))+
                                              [trajresid[0], trajresid[1]])

                #for before, after in window(joined_zranges, 2):
                #    trans_per_traj.append(list(before)+
                #                          list(after)+
                #                          list(trajresid))

        # Post-processing step to convert the defaultdict
        if return_stats and len(trans_count.items()) > 0:
            temp_counts = pd.DataFrame(trans_count.items(),
                                       columns=['Transition', 'Count'])
            temp_counts["TrajNum"], temp_counts["Resid"] = trajresid
            trans_counts_per_traj = trans_counts_per_traj.append(temp_counts,
                                                                 ignore_index=True)

    if return_stats:
        return trans_counts_per_traj
    else:
        colstrs = []
        for statenum in range(states_per_transition):
            colstrs.extend([transition_column+str(statenum)+"_Start",
                            transition_column+str(statenum)+"_Stop",
                            transition_column+str(statenum)+"_Dwell"])
        colstrs.extend(["TrajNum", "ResidID"])

        return pd.DataFrame(trans_per_traj, columns=colstrs)

        #return pd.DataFrame(trans_per_traj,
        #                    columns=[translabel+"_Start", translabel+"_Start", "Dwell_Start",
        #                             translabel+"_End", translabel+"_End", "Dwell_End",
        #                             "TrajNum", "ResidID"])

def macrostate_transitions_per_traj(coord_df,
                                    transition_column="OccMacrostate",
                                    dwell_cut=2,
                                    states_per_transition=2,
                                    return_stats=True):
    """
    Macrostates are defined (outside this function) in one or more
    transition columns. Rates are computed for this state stream using
    the a dwell time threshold in each state to satisfy a transition.
    Basically an identical copy of state_transitions_per_trajres without
    defining states by Z values on a 1D axis.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    transition_columns : list
        Column name(s) that define the macrostate at that timestep
    dwell_cut : int
        States less than dwell_cut timesteps are pruned.
    states_per_transition : int
        Number of states that make a transition, two state is default.
    return_stats : bool
        True returns count of transitions, False returns transitions
    """

    # Iterate over each resid in each trajectory
    classified_groups = coord_df.groupby(["TrajNum"])

    trans_counts_per_traj = pd.DataFrame()
    trans_per_traj = []

    # TODO: this contiguous time thing doesn't really apply since all
    # trajectories are continuous sequences of Frame numbers!
    for trajresid, trajresid_ts in classified_groups:
        # trajresid is a tuple of format (trajnum, resid)
        #print(trajresid),

        # Tiny bit of magic here, groupby consecutive time chunks:
        # http://stackoverflow.com/questions/26121668/slice-pandas-dataframe-in-groups-of-consecutive-values
        contig_timeblocks = trajresid_ts['Frame'] - np.arange(trajresid_ts.shape[0])

        for k, g in trajresid_ts.groupby(contig_timeblocks):
            trans_count = defaultdict(int)

            # Drop outside of pandas and use itertools to groupby zrange ID
            # until pandas supports this natively (Pandas 0.16 maybe?)
            grouped_iter = itertools.groupby(g[[transition_column,"Frame"]].values, key=lambda col: col[0])

            # Lose the iterator, now we have a list of format (zrangeid, (zrangeid, time))
            grouped_zranges = [(zid, list(dwell)) for zid, dwell in grouped_iter]

            # Preserve the first time label, sorry the indexing is so bad!
            # Now we have a list of format (zrangeid, start_time, dwell_time)
            condensed_zranges = [(zpair[0], zpair[1][0][1], len(zpair[1]))
                                 for zpair in grouped_zranges]

            '''
            # This code only cuts and merges only once

            # Remove short-lived states with a quick pruning.
            cutoff_zranges = [zpair for zpair in condensed_zranges if zpair[2] >= dwell_cut]

            # Now we stitch together pairs with identical zrange ID by
            # iteratively merging time blocks and adjusting the dwell time
            # of the first

            #print("CONDENSED: ", cutoff_zranges)
            joined_zranges = []
            if len(cutoff_zranges) > 1:
                # Iteratively pop off a contiguous timeblock from the list
                # and merge it with the next timeblock if it shares a common
                # ZRange
                curr_zrange = cutoff_zranges.pop(0)
                while len(cutoff_zranges) > 0:
                    next_zrange = cutoff_zranges.pop(0)
                    if curr_zrange[0] == next_zrange[0]:
                        curr_zrange = (curr_zrange[0], curr_zrange[1],
                                       sum(next_zrange[1:3])-curr_zrange[1])
                    else:
                        joined_zranges.append(curr_zrange)
                        curr_zrange = next_zrange
                joined_zranges.append(curr_zrange)

            '''

            for smaller_dwell_cut in range(2, dwell_cut+1):
                cutoff_zranges = [zpair for zpair in condensed_zranges if zpair[2] >= smaller_dwell_cut]

                joined_zranges = []
                if len(cutoff_zranges) > 1:
                    # Iteratively pop off a contiguous timeblock from the list
                    # and merge it with the next timeblock if it shares a common
                    # ZRange
                    curr_zrange = cutoff_zranges.pop(0)
                    while len(cutoff_zranges) > 0:
                        next_zrange = cutoff_zranges.pop(0)
                        if curr_zrange[0] == next_zrange[0]:
                            curr_zrange = (curr_zrange[0], curr_zrange[1],
                                           sum(next_zrange[1:3])-curr_zrange[1])
                        else:
                            joined_zranges.append(curr_zrange)
                            curr_zrange = next_zrange
                    joined_zranges.append(curr_zrange)

                #print(joined_zranges)
                condensed_zranges = joined_zranges

            #import pdb
            #pdb.set_trace()

            #print("JOINED: ",joined_zranges)
            # If you want stats, just increment a counter
            # If you want transitions, take the whole before-after pair
            if return_stats:
                for states in window(joined_zranges, states_per_transition):
                    state_trans_str = "-".join([str(state[0]) for state in states])
                    trans_count[state_trans_str] += 1

                #for before, after in window(joined_zranges, 2):
                #    trans_count[str(before[0])+"-"+str(after[0])]+=1
            else:
                for states in window(joined_zranges, states_per_transition):
                    # Just make sure we have all unique states in this set,
                    # otherwise, it's just a recrossing!
                    if len(set([state[0] for state in states])) == states_per_transition:
                        trans_per_traj.append(list(itertools.chain.from_iterable(states))+
                                              [trajresid])

                #for before, after in window(joined_zranges, 2):
                #    trans_per_traj.append(list(before)+
                #                          list(after)+
                #                          list(trajresid))

        # Post-processing step to convert the defaultdict
        if return_stats and len(trans_count.items()) > 0:
            temp_counts = pd.DataFrame(trans_count.items(),
                                       columns=['Transition', 'Count'])
            temp_counts["TrajNum"] = trajresid
            trans_counts_per_traj = trans_counts_per_traj.append(temp_counts,
                                                                 ignore_index=True)

    if return_stats:
        return trans_counts_per_traj
    else:
        colstrs = []
        for statenum in range(states_per_transition):
            colstrs.extend([transition_column+str(statenum)+"_Start",
                            transition_column+str(statenum)+"_Stop",
                            transition_column+str(statenum)+"_Dwell"])
        colstrs.extend(["TrajNum"])

        return pd.DataFrame(trans_per_traj, columns=colstrs)

        #return pd.DataFrame(trans_per_traj,
        #                    columns=[translabel+"_Start", translabel+"_Start", "Dwell_Start",
        #                             translabel+"_End", translabel+"_End", "Dwell_End",
        #                             "TrajNum", "ResidID"])

def state_transitions_per_trajres(coord_df, dividers, state_names=None,
                                  transition_axis="Z", dwell_cut=2,
                                  return_stats=True):
    """
    A state transition dictionary produces a count of transitions or
    transition rates for the solute defined in coord_df that was tracked
    in a geometric region of simulation as it traverses states defined
    by the dividers list. Note that left and right of the dividers still
    indicate states. In ion channel simulations, this is typically an
    ion traversing specific regions of the permeation pore in 1 dimension.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    dividers : list
        Z dimensional dividers that define states that a solute might traverse.
    transition_axis : str
        Column name in coord_df to apply dividers.
    dwell_cut : int
        States less than dwell_cut timesteps are pruned.
    return_stats : bool
        True returns count of transitions, False returns transitions
    """

    # Drop nuisance columns and digitize the transition_axis column
    # using the dividers list.
    classified_df = coord_df.loc[:,("TrajNum","Frame","ResidID",transition_axis)]

    digitized_axis = np.digitize(classified_df[transition_axis].values,
                                 bins=dividers)
    if state_names == None:
        classified_df["ZRange"] = pd.Series(digitized_axis,
                                            index=coord_df.index)
    else:
        classified_df["ZRange"] = pd.Series(digitized_axis,
                                            index=coord_df.index).map(pd.Series(state_names))


    # Iterate over each resid in each trajectory
    classified_groups = classified_df.groupby(["TrajNum","ResidID"])

    trans_counts_per_traj = pd.DataFrame()
    trans_per_traj = []

    for trajresid, trajresid_ts in classified_groups:
        # trajresid is a tuple of format (trajnum, resid)
        #print(trajresid),

        # Tiny bit of magic here, groupby consecutive time chunks:
        # http://stackoverflow.com/questions/26121668/slice-pandas-dataframe-in-groups-of-consecutive-values
        contig_timeblocks = trajresid_ts['Frame'] - np.arange(trajresid_ts.shape[0])

        for k, g in trajresid_ts.groupby(contig_timeblocks):
            trans_count = defaultdict(int)

            # Drop outside of pandas and use itertools to groupby zrange ID
            # until pandas supports this natively (Pandas 0.16 maybe?)
            grouped_iter = itertools.groupby(g[["ZRange","Frame"]].values, key=lambda col: col[0])

            # Lose the iterator, now we have a list of format (zrangeid, (zrangeid, time))
            grouped_zranges = [(zid, list(dwell)) for zid, dwell in grouped_iter]

            # Preserve the first time label, sorry the indexing is so bad!
            # Now we have a list of format (zrangeid, start_time, dwell_time)
            condensed_zranges = [(zpair[0], zpair[1][0][1], len(zpair[1]))
                                 for zpair in grouped_zranges]

            # Remove short-lived states with a quick pruning.
            cutoff_zranges = [zpair for zpair in condensed_zranges if zpair[2] >= dwell_cut]

            # Now we stitch together pairs with identical zrange ID by
            # iteratively merging time blocks and adjusting the dwell time
            # of the first

            #print("CONDENSED: ", cutoff_zranges)
            joined_zranges = []
            if len(cutoff_zranges) > 1:
                # Iteratively pop off a contiguous timeblock from the list
                # and merge it with the next timeblock if it shares a common
                # ZRange
                curr_zrange = cutoff_zranges.pop(0)
                while len(cutoff_zranges) > 0:
                    next_zrange = cutoff_zranges.pop(0)
                    if curr_zrange[0] == next_zrange[0]:
                        curr_zrange = (curr_zrange[0], curr_zrange[1],
                                       sum(next_zrange[1:3])-curr_zrange[1])
                    else:
                        joined_zranges.append(curr_zrange)
                        curr_zrange = next_zrange
                joined_zranges.append(curr_zrange)

            #print("JOINED: ",joined_zranges)
            # If you want stats, just increment a counter
            # If you want transitions, take the whole before-after pair
            if return_stats:
                for before, after in window(joined_zranges, 2):
                    trans_count[str(before[0])+"-"+str(after[0])]+=1
            else:
                for before, after in window(joined_zranges, 2):
                    trans_per_traj.append(list(before)+
                                          list(after)+
                                          list(trajresid))

        # Post-processing step to convert the defaultdict
        if return_stats and len(trans_count.items()) > 0:
            temp_counts = pd.DataFrame(trans_count.items(),
                                       columns=['Transition', 'Count'])
            temp_counts["TrajNum"], temp_counts["Resid"] = trajresid
            trans_counts_per_traj = trans_counts_per_traj.append(temp_counts,
                                                                 ignore_index=True)

    if return_stats:
        return trans_counts_per_traj
    else:
        return pd.DataFrame(trans_per_traj,
                            columns=["ZRange_Start", "Frame_Start", "Dwell_Start",
                                     "ZRange_End", "Frame_End", "Dwell_End",
                                     "TrajNum", "ResidID"])

def unique_state_transition_counts(coord_df, dividers, state_names=None,
                                   transition_axis="Z",
                                   dwell_cut=2, compute_rates=True):

    """
    A state transition dictionary produces a count of transitions or
    transition rates for the solute defined in coord_df that was tracked
    in a geometric region of simulation as it traverses states defined
    by the dividers list. This omits trajectory information and just gives you
    total counts of transitions for all unique transition types. Useful
    as one way of computing permeation rates.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    dividers : list
        Z dimensional dividers that define states that a solute might traverse.
    transition_axis : str
        Column name in coord_df to apply dividers.
    dwell_cut : int
        States less than dwell_cut timesteps are pruned.
    compute_rates : bool
        Normalizes the transition count by frame count in coord_df
    """
    trans = state_transitions_per_trajres(coord_df, dividers,
                                          state_names=state_names,
                                          transition_axis=transition_axis,
                                          dwell_cut=dwell_cut)

    # Compute total number of frames using the last recorded frame per trajectory
    if compute_rates:
        total_steps = sum(coord_df.groupby(["TrajNum"])["Time"].apply(lambda x: max(x.unique())))
        return trans.groupby("Transition")["Count"].sum()/total_steps
    else:
        return trans.groupby("Transition")["Count"].sum()

def mean_unique_state_transition_counts(coord_df, dividers, state_names=None,
                                        transition_axis="Z",
                                        dwell_cut=2, compute_rates=True):

    """
    A state transition dictionary produces a count of transitions or
    transition rates for the solute defined in coord_df that was tracked
    in a geometric region of simulation as it traverses states defined
    by the dividers list. This averages over trajectories and returns mean
    transition counts along with the standard error of mean. Useful as
    one way of computing permeation rates.

    Parameters
    ----------
    coord_df : DataFrame
        This is the master coordination dataframe of the project.
    dividers : list
        Z dimensional dividers that define states that a solute might traverse.
    transition_axis : str
        Column name in coord_df to apply dividers.
    dwell_cut : int
        States less than dwell_cut timesteps are pruned.
    compute_rates : bool
        Normalizes the transition count by frame count in coord_df
    """
    trans = state_transitions_per_trajres(coord_df, dividers,
                                          state_names=state_names,
                                          transition_axis=transition_axis,
                                          dwell_cut=dwell_cut)

    # Compute total number of frames using the last recorded frame per trajectory
    if compute_rates:
        total_steps = coord_df.groupby(["TrajNum"])["Time"].apply(lambda x: max(x.unique()))
        trans_normalized = trans.set_index("TrajNum")
        trans_normalized["Rate"] = trans_normalized["Count"].div(total_steps,
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
    return ggstats

def prune_time(df, start, end):
    return df[(df["Time"] >= start) & (df["Time"] <= end)]

if __name__ == '__main__':
    parser = ArgumentParser(
    description='This script extracts basic transition statistics of \
    coordination with attention to statistics across multiple timeseries.')
    parser.add_argument(
    '-c', dest='cfg_path', type=str, required=True,
    help='a configfile describing data paths and parameters for a project')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)
    #print(TestProject.coord_ts)

    coord_df_noequil = prune_time(TestProject.coord_ts['1st Shell Na+ SF Coordination'],
                                  TestProject.start_time,
                                  TestProject.end_time)

    #print(state_transitions_per_trajres(coord_df, dividers))
    #print(mean_unique_state_transition_counts(coord_df_noequil,
    #                                          TestProject.state_dividers,
    #                                          state_names=TestProject.state_names,
    #                                          compute_rates=True))

    #print(unique_state_transition_counts(coord_df_noequil,
    #                                     TestProject.state_dividers,
    #                                     state_names=TestProject.state_names,
    #                                     compute_rates=True))

    rates_per_dwell_time = []
    mean_rates_per_dwell_time = []
    sem_rates_per_dwell_time = []
    for dwell_cut in range(1,13):
        rates_per_dwell_time.append(unique_state_transition_counts(coord_df_noequil,
                                    TestProject.state_dividers,
                                    state_names=TestProject.state_names,
                                    dwell_cut=dwell_cut,
                                    compute_rates=True))

        rate_stats_per_dwell_time = mean_unique_state_transition_counts(coord_df_noequil,
                                              TestProject.state_dividers,
                                              state_names=TestProject.state_names,
                                              dwell_cut=dwell_cut,
                                              compute_rates=True)

        mean_rates_per_dwell_time.append(rate_stats_per_dwell_time["Mean"])
        sem_rates_per_dwell_time.append(rate_stats_per_dwell_time["SEM"])

    joint_rates_per_dwell = pd.concat(rates_per_dwell_time, axis=1)
    joint_rates_per_dwell.columns = [str(x)+" Dwell" for x in range(1,13)]

    joint_mean_rates_per_dwell = pd.concat(mean_rates_per_dwell_time, axis=1)
    joint_mean_rates_per_dwell.columns = [str(x)+" Dwell" for x in range(1,13)]

    joint_sem_rates_per_dwell = pd.concat(sem_rates_per_dwell_time, axis=1)
    joint_sem_rates_per_dwell.columns = [str(x)+" Dwell" for x in range(1,13)]

    writer = pd.ExcelWriter(TestProject.output_name+'_transitions.xlsx')
    joint_rates_per_dwell.to_excel(writer,'Total Rates')
    joint_mean_rates_per_dwell.to_excel(writer,'Mean Rates')
    joint_sem_rates_per_dwell.to_excel(writer,'SEM Rates')
    writer.save()
