# NGO_1D_sITG_equation
Code necessary to create the data and train a Neural Green's Operator for a surrogate model capable of finding the solutions to a simplified gyrokinetic equation.

In order to run this code all files must be kept in the same folder, after which the Eigen_data.py file can be used to create a data file. 

With this data generated the name to this file should be matched in the NGO.py file.

Given that all packages that are required have been installed the code can then be used to train a model. 

The NGO.py file also enables the switching of which model is used, being able to switch between a simple MLP, a DeepONet, an approximated NGO and an NGO based on a CNN.
