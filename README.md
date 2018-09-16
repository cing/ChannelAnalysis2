ChannelAnalysis2: Meta-analysis package for molecular dynamics data from ion channels
=========

After molecular dynamics simulations are performed, various analyses are
performed on the dataset using external applications like GROMACS tools,
MDAnalysis, MDTraj, VMD scripts, or countless scripts that perform advanced functionality.
Many simulation studies focus on the analysis
of single timeseries datasets. Due to the non-ergodic nature of molecular simulations,
even simulation runs starting from the same initial conditions will greatly diverge
during molecular dynamics simulations. In the case of ion channel simulations of the voltage-gated
sodium channel, single trajectories when considered in isolation, exhibit extremely
different ionic binding modes due to the "orthogonal degrees of freedom" to ion permeation.

A more statistically rigorous method would be to conduct multiple simulation repeats and perform
meta-analysis across this combined dataset. However, the complexity of data analysis and
plotting greatly increases in this scenario, due to the need to perform
averaging and standard deviation calculations over histograms and timeseries data.
To futher add complexity, when comparing multiple datasets of different systems (5 or more),
it is, again, non-trivial to perform this comparison manually and combine figures for publication.

The purpose of this package is to assist in automation of figure generation directly
from raw timeseries data extracted from molecular dynamics trajectories.
It is assumed that each dataset will consist of multiple simulation repeats. The file required
for input is a config script that describes the columns of the raw data and datatypes.

As this package is a successor to the project ChannelAnalysis, a great deal of functionality
is directed towards the study of ion coordination as a function of ion position along the
principle channel axis. This package depends strongly on the pandas package, loading all
data into dataframes that are exposed as members of the Project class.

Warning: this is not tested and may not work with newer versions of Pandas!
