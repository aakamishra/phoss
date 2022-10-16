from experiment_analysis import ExperimentAnalysis
from experiment_group import ExperimentGroup

egroup1 = ExperimentGroup('ASHA', None, [90, 100, 110], save_dir='checkpoints')
result1 = egroup1.run()

egroup2 = ExperimentGroup('Hyperband', None, [90, 100, 110],
                          save_dir='checkpoints')
result2 = egroup2.run()

# ExperimentAnalysis.plot_results([result1, result2])
