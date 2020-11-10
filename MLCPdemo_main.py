import pandas as pd
import datetime
import MLCPdemo_methods

# Specify parametres
mtypes = {"typeA": {'o': 24, 'v': 0.5}, "typeB": {'o': 48, 'v': 1}} # maintenance type definition
T = 7*24 # planning horizon, cannot exceed the validity of the input data (which is in this case 7 days = 7*24 hours)
eps = 0.001 # technical parameter, see MILP definition
L_D_max = 5 # number of maintenance locations opened during daytime at maximum
startday_hr = 7 # start of the daytime time window (hour of day)
startnight_hr = 19 # start of the nighttime time window (hour of day)

# Read MO table input, containing all MOs.
# Should contain a list of all MOs, with for each MO the trainnr, the MO start time s in hours after midnight of the first day,
# the MO end time e in hours after midnight of the first day, and the location l.
motbl_in = pd.read_csv("motbl.csv", index_col=0)

# Add some extra variables: the variable d indicates for each MO whether it is a daytime MO or not,
# and the s_time and e_time variables are timestamps converting the variables s and e to interpretable dates, assuming here the first day in the analysis is January 1st, 1970.
motbl_in["d"] = motbl_in.apply(lambda x: MLCPdemo_methods._moIsDuringDay(s=x.s, e=x.e, startday_hr=startday_hr, startnight_hr=startnight_hr), axis=1)
motbl_in["s_time"] = [pd.Timestamp("1970-01-01") + pd.Timedelta(row.s, unit="hours") for index,row in motbl_in.iterrows()]
motbl_in["e_time"] = [pd.Timestamp("1970-01-01") + pd.Timedelta(row.e, unit="hours") for index,row in motbl_in.iterrows()]

# Run MLCP model
time_start = datetime.datetime.now()
prob, dvs, inp, motbl, status = MLCPdemo_methods.run_MLCP(motbl_in, mtypes, T, eps, L_D_max)
running_time = datetime.datetime.now() - time_start

# Obtain output: the objective, the schedule (output_mos) and the opened locations (output_locs) in the final solution
output_mos = MLCPdemo_methods._getOutputMOs(motbl, prob, inp, dvs)
output_locs = MLCPdemo_methods._getOutputLocs(prob, inp, dvs)
print("MLCP model run successfully; final objective value = " + str(prob.objective.value()))