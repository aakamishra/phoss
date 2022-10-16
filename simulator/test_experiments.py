from experiment_analysis import ExperimentAnalysis
from experiment_group import ExperimentGroup

seeds = [90, 100, 110, 120, 130]
methods = ["Random", "PBT", "ASHA", "Hyperband"]
results = []
exp_groups = []
save_dir = "checkpoints"
num_workers = 8
num_samples = 30
max_num_epochs = 50

for method in methods:
    egroup = ExperimentGroup(method, 
                            None, 
                            seeds=seeds, 
                            max_num_epochs=max_num_epochs,
                            num_actors=num_workers,
                            num_samples=num_samples,
                            save_dir="checkpoints")
    result = egroup.run()
    exp_groups.append(egroup)
    results.append(result)

ExperimentAnalysis.plot_results(results)

