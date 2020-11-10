###############################
MLCP demonstration version 1.0
README
###############################

This set of scripts allow users to solve the Maintenance Location Choice Problem (MLCP) themselves.

Three files are provided.
- motbl.csv contains synthetic rolling stock circluation data for 7 days, for 20 rolling stock units.
Each line contains a table of Maintenance Opportunities (MOs), with associated rolling stock unit number (trainnr), start time s, end time e and location l.

- In MLCPdemo_main.py, the main parameters are set, the data is read, the model is run and some output is generated.

- MLCPdemo_methods contains all required methods used by MLCPdemo_main.py .
Specifically it defines the LP model using the PuLP package and solves it.
This file also is responsible for the actual solver call, which defaults to the standard PuLP solver but can be changed to another solver such as Gurobi, if available.

The following packages were used during the development process: numpy (version 1.19.3), pandas (version 1.1.4), pulp (version 2.3.1). The scripts were run with Python version 3.7.7 .