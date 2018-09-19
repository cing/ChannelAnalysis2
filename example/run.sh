# This entire suite of analysis scripts suddenly becomes a lot better when you have multiple simulation trajectories
# (you get errorbars, shaded standard error of mean regions, etc.)

config=input.cfg

# Produce an axial timeseries for ion permeation, showing red/green/blue ions, note: ions are not uniquely distinguishable
python ../scripts/ProduceTS_ordered.py -c ${config} -ts "1st Shell Na+ SF Coordination"

# Produce another axial timeseries for ion permeation, but this time show macrostate occupancy blocks underneath (based on a dwell time cutoff of 40)
python ../scripts/ProduceMacroRateEvents.py -c ${config} -ts "2nd Shell Na+ SF Coordination"

# Print a diagram of the primary SF microstates
python ../scripts/ProduceBindingSiteOrderings.py -c ${config} -ts "1st Shell Na+ SF Coordination" "2nd Shell Na+ SF Coordination" > NAVAB_XTAL_PD-NORES-0kcal-CHARMM_NONB_WT_SOD150_orderings.txt

# Make a classic 1D PMF along the channel axis
python ../scripts/ProducePMF_bulkfix.py -c ${config} -ts "1st Shell Na+ SF Coordination Extended"

# For ion occupancy states, print an axial histogram of ions within the SF with their respective binding modes
python ../scripts/ProduceOccSplit_ModeHistograms.py -c ${x} -ts "1st Shell Na+ SF Coordination" "2nd Shell Na+ SF Coordination"

# For ion occupancy states, print an axial histogram for each ion (ordered red, green, blue), note: ions are not uniquely distinguishable
python ../scripts/ProduceOrderPrimeHistograms.py -c ${config} -ts "2nd Shell Na+ SF Coordination"

# Compute binding site occupancy bar charts for positions within the SF and save numerical results to a file
python ../scripts/ProduceModeOccupancyBarChartsOPLS_Joint.py -c ${config} -ts "1st Shell Na+ SF Coordination" "2nd Shell Na+ SF Coordination" > NAVAB_XTAL_PD-NORES-0kcal-CHARMM_NONB_WT_SOD150_J_modeoccupancy.txt

# Compute some occupancy state kinetics (where transitions are defined by occupancy in states for more than 250 steps)
python ../scripts/ProduceMacroRatesVsDwell.py -c ${config} -ts "2nd Shell Na+ SF Coordination" > NAVAB_XTAL_PD-NORES-0kcal-CHARMM_NONB_WT_SOD150_total_occ_macrorates_dwell250.txt

# For the E177 sidechain dihedral angles, plot the X1 vs. X2 heatmap
python ../scripts/ProduceChi1Chi2.py -c ${config} -ts "E177 Chi1-Chi2 Dihedral Angles"

# Examine the first solvation shell ligands of ions traversing the SF along the axial coordinate
python ../scripts/ProduceAvgCoordHistograms.py -c ${config} -ts "1st Shell Na+ SF Oxy Coordination"

# Produce a 2D histogram of outer vs. inner ions for all occupancy states
python ../scripts/Produce2DPMF_bulkfix.py -c ${config} -ts "1st Shell Na+ SF Coordination Extended"

# For ion occupancy states, produce a 2D histogram of outer vs inner for all states
python ../scripts/ProduceOcc_2DPMFs.py -c ${config} -ts "2nd Shell Na+ SF Coordination"

# Plot the axial distribution of channel ligand oxygen atoms in the SF
python ../scripts/ProduceOxygenHistograms.py -c ${config}

# Produce a plot of the channel occupancy versus time
python ../scripts/ProduceChannelOccVsTime.py -c ${config} -ts "2nd Shell Na+ SF Coordination"
