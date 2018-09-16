""" This script populates the "Project" class with data described in the
configuration file argument. It loads text data into pandas dataframes,
handles pickling and unpickling of this binary data in addition to
loading legacy datatypes that were used in previous iterations of
this codebase """

from __future__ import print_function, division, absolute_import

import math
import pandas as pd
import numpy as np
from ChannelAnalysis2.ConvertMultiColumn import parse_legacy_datatype
from sys import argv
from configparser import ConfigParser
from argparse import ArgumentParser
from glob import glob
from string import ascii_uppercase
from re import compile
from itertools import groupby

# TODO: Ugh, refactor
def unit_vector(data):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis."""
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
        data /= math.sqrt(np.dot(data, data))
        return data
    length = np.atleast_1d(np.sum(data * data, None))
    np.sqrt(length, length)
    data /= length
    return data

def rotation_matrix(angle, direction):
    """Return matrix to rotate about axis using angle/direction."""
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(
        (   (cosa, 0.0, 0.0),
            (0.0, cosa, 0.0),
            (0.0, 0.0, cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        ((0.0, -direction[2], direction[1]),
        (direction[2], 0.0, -direction[0]),
        (-direction[1], direction[0], 0.0)), dtype=np.float64)
    return R.T

class Project(object):
    """ChannelAnalysis2 Project

    A project is a container for a group of dataframes pertaining to one
    set of simulations. It's the python embodiment of the config file,
    and is how you access project data. There are plotting routines
    that can plot multiple projects.

    Parameters
    ----------
    cfg_path : str
        The filepath to the configuration script that describes this data
    coord_ts : list
        Optional list of coordinate timeseries to add (to speed up loading)
    coord_ts : list
        Optional list of auxiliary timeseries to add (to speed up loading)

    Attributes
    ----------
    `components_` : array-like, shape (n_components, n_features)
        Components with maximum autocorrelation.
    `offset_correlation_` : array-like, shape (n_features, n_features)
    `eigenvalues_` : array-like, shape (n_features,)
    `eigenvectors_` : array-like, shape (n_components, n_features)
    `means_` : array, shape (n_features,)
    `n_observations` : int
    `n_sequences` : int
    """


    # When you initialize a new project, it attempts to extract all the data
    # it can from the configuration file to population the data structure.
    def __init__(self, cfg_path, coord_ts=None, aux_ts=None):
        cfg = ConfigParser()
        cfg.read(cfg_path)

        # TODO: You can't specify coord_ts AND aux_ts, failure.
        if "global_config" in cfg:
            self._load_globals(cfg["global_config"])

            # Before loading any coordinate timeseries, load
            # principal axis tilt vectors if they exist!
            if self.correct_tilt != 0:
                self._load_paxis(cfg)

            if coord_ts != None:
                self.coord_ts = {}
                for ts_name in coord_ts:
                    self._load_coordts(cfg, ts_name)
            elif aux_ts != None:
                self.aux_ts = {}
                for ts_name in aux_ts:
                    self._load_auxts(cfg, ts_name)
            else:
                self._load_timeseries(cfg)
        else:
            print("Unable to find global_config in %s" % cfg_path)

    # TODO: Unify individual loads and group load
    def _load_coordts(self, cfg, ts_name, max_config_entries=10):
        """
        Coord timeseries load data constructor for specific timeseries
        """
        for config_num in range(max_config_entries):
            if 'coord_config'+str(config_num) in cfg:
                if cfg["coord_config"+str(config_num)].get("ts_name") == ts_name:
                    tsname, color, ts = self._process_timeseries(cfg["coord_config"+str(config_num)],
                                                         convert_col_factor=self.convert_col_factor,
                                                         coltype="coordination",
                                                         up_orientation=self.up_orientation)
                    self.coord_ts[tsname] = ts
                    self.color_ts[tsname] = color

    def _load_auxts(self, cfg, ts_name, max_config_entries=10):
        """
        Aux timeseries load data constructor for specific timeseries
        """
        for config_num in range(max_config_entries):
            if 'aux_config'+str(config_num) in cfg:
                if cfg["aux_config"+str(config_num)].get("ts_name") == ts_name:
                    tsname, color, ts = self._process_timeseries(cfg["aux_config"+str(config_num)],
                                                         convert_col_factor=self.convert_col_factor,
                                                         up_orientation=self.up_orientation)
                    self.aux_ts[tsname] = ts
                    self.color_ts[tsname] = color

    def _load_paxis(self, cfg):
        """
        Aux timeseries load data constructor for paxis data only!
        """
        if 'paxis_config' in cfg:
            tsname, color, ts = self._process_timeseries(cfg["paxis_config"],
                                                 convert_col_factor=self.convert_col_factor,
                                                 coltype="paxis",
                                                 up_orientation=self.up_orientation)
            #self.paxis_raw = {}
            #self.paxis_raw[tsname] = ts
            self.paxis_df = self._compute_rotation_matrices(ts)
        else:
            print("Unable to find paxis_config and correct_tilt is set")

    def _load_timeseries(self, cfg, max_config_entries=10):
        """
        Timeseries load data constructor for all available data
        """
        self.coord_ts = {}
        for config_num in range(max_config_entries):
            if 'coord_config'+str(config_num) in cfg:
                tsname, color, ts = self._process_timeseries(cfg["coord_config"+str(config_num)],
                                                     convert_col_factor=self.convert_col_factor,
                                                     coltype="coordination",
                                                     up_orientation=self.up_orientation)
                print("- Loaded: "+tsname)
                self.coord_ts[tsname] = ts
                self.color_ts[tsname] = color


        self.resid_ts = {}
        for config_num in range(max_config_entries):
            if 'resid_config'+str(config_num) in cfg:
                tsname, color, ts = self._process_timeseries(cfg["resid_config"+str(config_num)],
                                                     convert_col_factor=self.convert_col_factor,
                                                     up_orientation=self.up_orientation)
                print("- Loaded: "+tsname)
                self.resid_ts[tsname] = ts
                self.color_ts[tsname] = color

        self.dihe_ts = {}
        for config_num in range(max_config_entries):
            if 'dihe_config'+str(config_num) in cfg:
                tsname, color, ts = self._process_timeseries(cfg["dihe_config"+str(config_num)],
                                                     convert_col_factor=self.convert_col_factor,
                                                     coltype="dihedral")
                print("- Loaded: "+tsname)
                self.dihe_ts[tsname] = ts
                self.color_ts[tsname] = color

        self.aux_ts = {}
        for config_num in range(max_config_entries):
            if 'aux_config'+str(config_num) in cfg:
                tsname, color, ts = self._process_timeseries(cfg["aux_config"+str(config_num)],
                                                     convert_col_factor=self.convert_col_factor,
                                                     up_orientation=self.up_orientation)
                print("- Loaded: "+tsname)
                self.aux_ts[tsname] = ts
                self.color_ts[tsname] = color

        '''
        self.mode_ts = {}
        for config_num in range(max_config_entries):
            if 'mode_config'+str(config_num) in cfg:
                tsname, ts = self._process_timeseries(cfg["mode_config"+str(config_num)])
                self.mode_ts[tsname] = ts
        '''


    def _compute_rotation_matrices(self, df, pax_regex="P3-", box_vector=[0,0,1]):
        """
        Takes a dataframe containing principal axis data and returns rotation
        vectors needed to align coordinates along component specificed
        by "vector_regex". Returns tuple of angular deviation (N_frames x 1)
        and the transpose of the rotation matrix (ready for dot product)
        for all frames (array dimensions N_frames x 3 x 3)
        """
        all_p3=df.filter(regex=pax_regex).values

        # Get angular deviation and correct for flipped vectors
        angle_dev = np.arccos(np.dot(all_p3, np.array(box_vector)))

        # Principal axis can flip 180 sometimes, so correct for that
        vec2invert = angle_dev > 0.5*math.pi
        angle_dev[vec2invert] = math.pi - angle_dev[vec2invert]
        all_p3[vec2invert] *= -1
        
        rotvecs = np.cross(all_p3,np.array(box_vector))
        rotvecs /= np.expand_dims(np.linalg.norm(rotvecs,axis=1),1)
        rotmats = [rotation_matrix(a,rv) for a,rv in zip(angle_dev, rotvecs)]

        # Test if the rotation vector applied to the raw principal
        # axis data is truly aligned along the box vector [0,0,1]
        #rotmats_stack = np.stack(rotmats,axis=0)
        #assert np.all(np.einsum('ij,ijk->ik', all_p3, rotmats_stack)[:,2] - 1 < 0.0001)
        
        angle_df = df[["Frame","TrajNum"]].copy()
        angle_df["Deviation"]=angle_dev
        angle_df["RotationMatrix"]=rotmats #_stack.tolist()

        return angle_df

    def _load_globals(self, cfg):
        """
        Global data stores project metadata that describes the entire project.
        """
        self.name = cfg.get("project_name","Protein Simulation")
        self.output_name = cfg.get("output_name","NAVAB_")
        self.color_ts = {}

        self.dt = float(cfg.get("dt","0.020"))
        self.start_time = float(cfg.get('start_time',0))
        self.end_time = float(cfg.get('end_time',1000))
        self.up_orientation = bool(int(cfg.get('up_orientation', "1")))
        self.correct_tilt = bool(int(cfg.get('correct_tilt', "0")))
        self.convert_col_factor = int(cfg.get('convert_col_factor', "0"))

        # Default is 100% arbitrary, no choice really!
        self.state_dividers = [float(div) for div in cfg.get('state_dividers',
                               ",".join([str(x) for x in np.arange(-1.0,1.5,0.5)])).split(",")]

        # State dividers act on coordinate columns so it's best to multiply those
        if self.convert_col_factor != 0:
            self.state_dividers = [self.convert_col_factor*div for div in self.state_dividers]

        self.state_names = cfg.get('state_names',
                                   ",".join(ascii_uppercase[:len(self.state_dividers)-1])).split(",")

        # Lastly, if orientation is flipped, invert the state divider values.
        # On second thought, the dividers should stay the same just the Z axis flipped!
        #if not self.up_orientation:
        #    self.state_dividers = list(np.array(self.state_dividers)*-1)

    def _process_timeseries(self, cfg, convert_col_factor=0, coltype="coordination",
                            up_orientation=True, orientation_column="Z"):
        '''
        Here's a function that processes coordination timeseries data, it
        is not applicable to some other timeseries files like dihedral angles
        '''

        # Multiple glob strings are accepted, so iterate over those.
        filepaths = []
        for path in cfg["filepath"].split():
            filenames = glob(path)
            if filenames:
                filepaths.extend(filenames)
            else:
                raise Exception("Unable to open coordination data at location: %s" % path)

        # Column headers must be specified, or else you fail at data management.
        col_headers = cfg.get("col_headers").split(",")
        max_cols = len(col_headers)
        if max_cols == 0:
            raise Exception("No column headers provided, cannot read input.")

        if len(filepaths) > 0:
            #TODO: This parse legacy thing is only temporary, and all files should be phased
            # to a pickled dataframe type
            raw_data = parse_legacy_datatype(filepaths,cols_per_entry=max_cols)
            rlen = len(raw_data[0])
        else:
            raise Exception("Unable to open any coordination data, check path.")

        # If they didn't supply data types, just float them all... =(
        col_dtypes = cfg.get("col_dtypes",",".join(["float"]*rlen)).split(",")
        timeseries_df = pd.DataFrame(raw_data, columns=col_headers)

        timeseries_df['RowID'] = pd.Series(cfg.get("extra_identifier","None"),
                                            index=timeseries_df.index)

        # Do some data type extraction from the config file and convert the types
        # TODO: Make this general to all types, not just float, int, str
        float_names = [col_headers[i] for i, dt in enumerate(col_dtypes) if dt == "float" or dt =="int"]
        int_names = [col_headers[i] for i, dt in enumerate(col_dtypes) if dt == "int"]
        str_names = [col_headers[i] for i, dt in enumerate(col_dtypes) if dt == "object"]

        timeseries_df[float_names] = timeseries_df[float_names].astype(float)
        timeseries_df[int_names] = timeseries_df[int_names].astype(int)
        timeseries_df[str_names] = timeseries_df[str_names].astype(object)

        # There's a chance that _load_globals has not run yet, but hopefully it has!
        try:
            timeseries_df["Time"] = timeseries_df["Frame"]*self.dt
        except:
            timeseries_df["Time"] = timeseries_df["Frame"]*1.0

        # Here we'll apply the principal axis rotation vector on each
        # coordinate column in sets of X-Y-Z (with a common prefix)
        if self.correct_tilt and coltype != "dihedral" and coltype != "paxis":
            # Maybe the rotation vectors have not been set?
            if True:
                # Extract all the coordinate columns (X,Y,Z)
                cs=list(timeseries_df.columns)
                df=pd.merge(timeseries_df, self.paxis_df, on=["Frame","TrajNum"])

                # TODO: No support for X-Y-Z and *-X, *-Y, *-Z in the same dataframe
                coordcols=[m.group(0) for l in cs for m in [compile("^[XYZ]$").search(l)] if m]
                if len(coordcols) == 3:
                    colnames = ["X","Y","Z"]
                    vecs = df[colnames].values
                    if vecs.shape[1] == 3:
                        rotmats=np.stack(df["RotationMatrix"].values, axis=0)
                        updated_coords = np.einsum('ij,ijk->ik', vecs, rotmats)
                        timeseries_df[colnames] = pd.DataFrame(updated_coords, columns=colnames)
                elif len(coordcols) == 0:
                    coordcols=[m.group(0) for l in cs for m in [compile("^.*-[XYZ]$").search(l)] if m]
                    # Apply rotation matrix to each group of X,Y,Z columns detected.
                    for k,v in groupby(coordcols, key=lambda x:x[:-2]):
                        colnames = list(v)
                        vecs = df[colnames].values
                        if len(colnames) == 3:
                            rotmats=np.stack(df["RotationMatrix"].values, axis=0)
                            updated_coords = np.einsum('ij,ijk->ik', vecs, rotmats)
                            timeseries_df[colnames] = pd.DataFrame(updated_coords, columns=colnames)
                else:
                    print("Dataframe did not contain coordinate tuples X,Y,Z")
            else:
                print("Unable to apply paxis tilt data, no rotation performed")

        # If there's a conversion factor, multiply all columns by it,
        # useful for rapid conversion from nm to angstrom
        convert_cols = cfg.get("convert_cols",None)
        if convert_cols != None:
            convert_cols_s = convert_cols.split(",")
            if coltype == "coordination":
                timeseries_df[convert_cols_s] *= convert_col_factor
            elif coltype == "dihedral":
                # Converts the angle range to 0-360 degrees
                #timeseries_df[convert_cols_s] *= -1 # Chakrabarti 2013, no longer used
                shifted_angles = 360.0*(timeseries_df[convert_cols_s] < 0).astype(int)
                timeseries_df[convert_cols_s] += shifted_angles

        # Invert the Z values if the protein is upside down!
        if not up_orientation:
            # If the column header contains orientation_column string, flip it!
            subts = timeseries_df.filter(regex=orientation_column)
            timeseries_df[subts.columns] = subts*-1
            #timeseries_df[orientation_column] *= -1

        return (cfg.get("ts_name",cfg.name),
                "#"+cfg.get("color_hex","000000"),
                timeseries_df)

    def total_simulation_time(self, coord_df, traj_detail=False, remove_equilibration=True):
        '''
        Computes the simulation time in coord_df (note that this can be any loaded
        timeseries inside Project, not just coordination. Returns total simulation data in ns.
        '''
        ts_totals = {}
        for name,df in coord_df.iteritems():
            if remove_equilibration:
                #print("Removed "+str(self.start_time)+" from "+str(df["TrajNum"].nunique())+" repeats.")
                lens = df[df["Time"] > self.start_time].groupby(["TrajNum"])["Time"].nunique()*self.dt
            else:
                lens = df.groupby(["TrajNum"])["Time"].nunique()*TestProject.dt

            if traj_detail:
                ts_totals[name] = lens.values
            else:
                ts_totals[name] = lens.sum()

        if np.all(np.vstack(ts_totals.values()),axis=0):
            print("bla")
        return ts_totals


if __name__ == '__main__':
    parser = ArgumentParser(
    description='Initializes a Project, encapsulating all trajectory \
    timeseries data, converting everything to dataframes.')
    parser.add_argument(
    '-c', dest='cfg_path', type=str, required=True,
    help='a configfile describing data paths and parameters for a project')
    args = parser.parse_args()

    # This builds a Project using the configuration file argument
    TestProject = Project(args.cfg_path)
    print("Successfully Loaded Project: %s" % TestProject.name)
    #print(TestProject.coord_ts)
