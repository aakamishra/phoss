# PHOSS

PHOSS is an open source Python hyperparameter search simulation platform for
distributed testing of scheduling algorithms.

## Installing from PyPI

To install PHOSS, run
```bash
$ pip install phoss
```

## Usage

To use PHOSS, just declare an ExperimentGroup with the arguments you want and
call `ExperimentGroup.run()`. This method returns a results object, which can
then be passed in as part of a list to the `ExperimentAnalysis.plot_results()`
method.

We have included two example usage scripts that use PHOSS:
- `test_experiments.py` creates and runs multiple `ExperimentGroup`s and plots
their results using `ExperimentAnalysis`.
- `test_landscaper.py` specifies a normal loss decay landscape and plots the
landscape generated by our simulator.

## Architecture Overview

In this section, we provide an overview of the core modules of PHOSS.

### Loss Landscaping

`Landscaper` handles the logic of generating a synthetic loss landscape based on
a user-specified distribution and workload configuration. These parameters can
be specified a JSON file that is passed into the `Landscaper` constructor.

### Experiment Runner

`ExperimentRunner` defines and runs a single experiment, which is performing a
hyperparameter search with a given scheduler and environment configurations. It
calls the `RayRunner` class which in turn calls Ray Tune to perform distributed
hyperparameter tuning.

PHOSS exposes the `ExperimentRunner.call_simulator()` method, which accepts a
list of configuration parameters. This method returns the results of the
specified experiment as a `Checkpoint` object and writes it to a JSON file too.

### Experiment Group

We define an experiment group to be 1 or more experiments varying only in their
random seed values. This class contains methods and the high-level logic for
conducting experiments across the entire group and combining their results.

At every epoch, PHOSS runs each individual experiment with `ExperimentRunner`
and averages over their results. PHOSS calculates the mean and cumulative best
arm regrets along with the average moving loss at each epoch.

All user-defined configurations for an `ExperimentGroup` are passed in as
constructor arguments. To run all experiments, call the `ExperimentGroup.run()`
method on the created object. `ExperimentGroup.run()` returns an
`ExperimentGroupResults` object.

### Experiment Analysis

`ExperimentAnalysis` provides a set of methods that visualize the graphs of
different experiment groups within the same plot, completing the end-to-end
flow of providing users with an easy way to compare multiple different
schedulers across multiple different configurations.

`ExperimentAnalysis.plot_results()` accepts a list of `ExperimentGroupResults`
as input.

## License

[Apache License 2.0](LICENSE)
