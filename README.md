# Multi-objective optimization of hydrostatic transmission performance with NSGA-II

The code in this repository was used to implement and visualize the results of the multi-objective optimization (MOO) of the hydrostatic transmission (HST) with the non-dominated sorting genetic algorithm (NSGA-II) realized with [pymoo](https://pymoo.org/) library. The images folder contains the graphs produced by the `plot_*` methods and functions, which allows for show, save, and show and save options. 

## HST analysis

The `hst_analysis` module demonstrates the use of the `HST` and `Regressor` objects adapted from the [effmap](https://github.com/ivanokhotnikov/effmap). The `HST` object is used to initialize and calculate the efficiency and performance metrics of the transmission. The `Regressor` is used to fit the exponential regression model to the machine speed data, and the linear regression model to the machine mass data. The `models` folder contains the saved, fir regression models.

## MOOHST

The `moohst` module is the main script to run the optimization, to plot the key metrics of the algorithm, its resultant Pareto front and set, its convergence. 