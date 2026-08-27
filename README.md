# Neural Green's Operator for a 1D sITG Equation

This repository contains a Python/PyTorch implementation of a Neural Green's
Operator (NGO) used to construct a surrogate model for a simplified 1D version of the gyrokinetic equations.

The project was developed as part of my MSc thesis in Science and Technology
of Nuclear Fusion at Eindhoven University of Technology (TU/e), in
collaboration with DIFFER.

The main objective is to investigate whether a neural operator, in this case specifically the NGO, can learn the
mapping between the parameters of a simplified gyrokinetic equation and its
solution, providing a route towards a computationally cheaper alternative for the solving of 5d gyrokinetic equation
in the future.

## Project overview

The code implements an NGO, as well as MLP and DeepONet, for the solving of a 1d drift-kinetic equation.
In this version of the equations an instability can arise called the sITG instability, thus the solution can either grow exponentially or be stable.

The workflow consists of:

1. Generating numerical solutions of the equation for different input
   parameters.
2. Using these solutions to construct a training dataset.
3. Training neural-network-based surrogate models to approximate the mapping
   from input parameters to the equation's solution.
4. Comparing the predictions of the surrogate models with the numerical
   reference solutions.

Several model architectures are implemented to provide comparisons between
different approaches:

- Simple direct neural network (MLP)
- DeepONet
- Approximated-matrix Neural Green's Operator
- CNN-based Neural Green's Operator

## Repository structure

### `Eigen_data.py`

Generates the numerical reference data used to train the surrogate models.

The generated data contains solutions of the simplified equation coupled
to different input parameters and initial conditions.

### `NGO.py`

Contains the training workflow.

The file allows different model architectures to be selected and trained,
including the MLP, DeepONet, approximated-matrix NGO and CNN-based NGO.

## Installation

The code has been developed and tested using **Python 3.10**.

All required packages are provided in 'requirements.txt'

## Results

The models trained using the training workflow can then be analysed using 'Model_comparison_hollow.py'.
Using this code plots are created comparing the growthrate, frequency and eigenvector predictions of numerical models to that of the ground truth. 

One of these results as generated is shown below for the growthrate give a specific set of inputs. This result shows that, for these inputs, the produced models with an MLP and a DeepONet produced low quality results while the NGO matched the ground truth nearly perfectly. 
![alt text](https://github.com/linwolter/NGO_1D_sITG_equation/blob/main/raw_growth_rate_vs_omega_n_50_1.0_1.5_3000_presentation_v2_allmodels.png?raw=true)
