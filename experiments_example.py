from phoss.experiment_analysis import ExperimentAnalysis
from phoss.experiment_group import ExperimentGroup

seeds = [161, 165]
methods = ['Random', 'ASHA', 'Hyperband', 'PBT']
results = []
exp_groups = []
save_dir = 'checkpoints'
num_workers = 16
num_samples = 100
max_num_epochs = 25

for method in methods:
    egroup = ExperimentGroup(
        method,
        None,
        seeds=seeds,
        max_num_epochs=max_num_epochs,
        num_actors=num_workers,
        num_samples=num_samples,
        save_dir='checkpoints',
        verbose=0
    )
    result = egroup.run()
    exp_groups.append(egroup)
    results.append(result)

ExperimentAnalysis.plot_results(results, show=False)

