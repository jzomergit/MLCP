###############################
MLCP demonstration version 1.0
README
###############################

This set of scripts allow users to solve the Maintenance Location Choice Problem (MLCP) themselves.

Three files are provided.
- motbl.csv contains synthetic rolling stock circluation data for 7 days, for 30 rolling stock units.
Each line contains a table of Maintenance Opportunities (MOs), with associated rolling stock unit number (trainnr), start time s, end time e and location l.

- In main_MLCPdemo.py, the main parameters are set, the data is read, the model is run and some output is generated.

- MLCP.py contains all required methods used by main_MLCPdemo.py .
Specifically it defines the LP model using the PuLP package and solves it.
This file also is responsible for the actual solver call, which defaults to the standard PuLP solver but can be changed to another solver such as Gurobi, if available. Solving using Gurobi yields significant runtime performance improvements and is therefore highly recommended.

The following packages were used during the development process: numpy (version 1.19.3), pandas (version 1.1.4), pulp (version 2.3.1). The scripts were run with Python version 3.7.7 .