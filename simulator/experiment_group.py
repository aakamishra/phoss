import numpy as np
import pandas as pd
from typing import Any, List, Tuple, Union
from experiment_runner import ExperimentRunner


DEFAULT_SIMULATOR_CONFIG = "simulator_configs/default_config.json"
DEFAULT_SCHEDULER_CONFIG = "scheduler_configs/default_config.json"
SCHEDULER_CONFIG_NAMES = ["ASHA", "Hyperband", "PBT", "PredASHA"]


class ExperimentGroup:

    # experiment_configs takes in a list of tuples. Each tuple contains:
    # 1. A string for a preexisting scheduler name
    # 2. Seed
    # 3. Optional scheduler object (if name is "custom")
    def __init__(
        self,
        experiment_configs: List[Tuple[str, int, Any]],
        num_samples: int = 16,
        max_num_epochs: int = 10,
        gpus_per_trial: int = 0,
        cpus_per_trial: int = 1,
        num_actors: int = 4,
        simulator_config: str = DEFAULT_SIMULATOR_CONFIG,
        scheduler_config: str = DEFAULT_SCHEDULER_CONFIG,
        verbose: int = 0,
        save_dir: str = '',
    ):
        self.experiment_configs = experiment_configs
        self.num_samples = num_samples
        self.max_num_epochs = max_num_epochs
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.num_actors = num_actors
        self.simulator_config = simulator_config
        self.scheduler_config = scheduler_config
        self.verbose = verbose
        self.save_dir = save_dir

    def _validate_experiment_configs(self) -> bool:
        print(self.experiment_configs)
        for scheduler_name, _, scheduler_obj in self.experiment_configs:
            if scheduler_name.lower() == 'custom' and not scheduler_obj:
                print('Custom scheduler specified but object not passed in!')
                return False
            elif scheduler_name not in SCHEDULER_CONFIG_NAMES:
                print('Could not find scheduler {}!'.format(scheduler_name))
                return False
        return True

    def _calculate_regret(
        self, true_file: str, data_file: str
    ) -> Tuple[List[float], List[float]]:
        """Get average and cumulative regret"""
        avgs = []
        sums = []
        running_sum = 0

        true_loss = np.genfromtxt(true_file, delimiter=',')
        data = data_file = pd.read_csv(data_file)

        for i in range(1, self.max_num_epochs):
            indices = data[data['training_iteration'] == i]['index'].unique()
            values = true_loss[i, indices]
            best_arm_sum = np.sum(sorted(true_loss[i])[:self.num_actors])
            avg = (np.sum(values) - (len(values) / self.num_actors) * best_arm_sum) / len(values)
            avgs.append(avg)
            running_sum += avg
            sums.append(running_sum)
        return [avgs, sums]

    # TODO: Make this a common function
    def _average_n_lists(self, lists: List[Union[int, float]]) -> List[float]:
        if len(lists) < 2:
            return lists
        n = len(lists[0])
        averages = []
        for i in range(1, len(lists)):
            if len(lists[i]) != n:
                print('Lists must have same size!')
                return []

        for i in range(n):
            sublst = []
            for lst in lists:
                sublst.append(lst[i])
            averages.append(sum(sublst) / len(sublst))
        return averages

    def run(self) -> None:
        if not self._validate_experiment_configs():
            return

        # Run all individual experiments
        checkpoints = []
        for scheduler_name, seed, scheduler_obj in self.experiment_configs:
            checkpoint = ExperimentRunner.call_simulator(
                scheduler_name,
                num_samples=self.num_samples,
                max_num_epochs=self.max_num_epochs,
                gpus_per_trial=self.gpus_per_trial,
                cpus_per_trial=self.cpus_per_trial,
                num_actors=self.num_actors,
                scheduler_object=scheduler_obj,
                simulator_config=self.simulator_config,
                scheduler_config=self.scheduler_config,
                seed=seed,
                verbose=self.verbose,
                save_dir=self.save_dir,
            )
            if not checkpoint:
                print('Simulator failed for config with name {} and seed {}. '
                      'Skipping!'.format(scheduler_name, seed))
            else:
                checkpoints.append(checkpoint)

        # Average the results if there are multiple checkpoints
        if not checkpoints:
            print('No available checkpoint objects!')
            return
        else:
            # Average individual regrets
            mean_regrets = []
            cumulative_regrets = []
            for checkpoint in checkpoints:
                mean_regret, cumulative_regret = self._calculate_regret(
                    checkpoint.true_sim_file, checkpoint.data_file
                )
                mean_regrets.append(mean_regret)
                cumulative_regrets.append(cumulative_regret)
            print('Mean regrets', self._average_n_lists(mean_regrets))
            print('Cumlative regrets', self._average_n_lists(cumulative_regrets))


if __name__ == '__main__':
    egroup = ExperimentGroup([('ASHA', 109, None)])
    egroup.run()
