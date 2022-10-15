import numpy as np
import pandas as pd
from typing import Any, List, Optional, Tuple, Union
from experiment_runner import ExperimentRunner


DEFAULT_SIMULATOR_CONFIG = "simulator_configs/default_config.json"
DEFAULT_SCHEDULER_CONFIG = "scheduler_configs/default_config.json"
SCHEDULER_CONFIG_NAMES = ["ASHA", "Hyperband", "PBT", "PredASHA"]


class ExperimentGroupResults:

    def __init__(
        self,
        scheduler_name: str,
        group_size: int,
        avg_mean_regret: List[float],
        avg_cumulative_regret: List[float],
    ):
        self.scheduler_name = scheduler_name
        self.group_size = group_size
        self.avg_mean_regret = avg_mean_regret
        self.avg_cumulative_regret = avg_cumulative_regret


class ExperimentGroup:

    def __init__(
        self,
        scheduler_name: int,
        scheduler_obj: Any,
        seeds: List[int],
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
        self.scheduler_name = scheduler_name
        self.scheduler_obj = scheduler_obj
        if not self._validate_scheduler_config():
            return
        self.seeds = seeds
        self.num_samples = num_samples
        self.max_num_epochs = max_num_epochs
        self.gpus_per_trial = gpus_per_trial
        self.cpus_per_trial = cpus_per_trial
        self.num_actors = num_actors
        self.simulator_config = simulator_config
        self.scheduler_config = scheduler_config
        self.verbose = verbose
        self.save_dir = save_dir

    def _validate_scheduler_config(self) -> bool:
        if self.scheduler_name.lower() == 'custom' and not self.scheduler_obj:
            print('Custom scheduler specified but object not passed in!')
            return False
        elif self.scheduler_name not in SCHEDULER_CONFIG_NAMES:
            print('Could not find scheduler {}!'.format(self.scheduler_name))
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

    def run(self) -> Optional[ExperimentGroupResults]:
        # Run all individual experiments
        checkpoints = []
        for seed in self.seeds:
            checkpoint = ExperimentRunner.call_simulator(
                self.scheduler_name,
                num_samples=self.num_samples,
                max_num_epochs=self.max_num_epochs,
                gpus_per_trial=self.gpus_per_trial,
                cpus_per_trial=self.cpus_per_trial,
                num_actors=self.num_actors,
                scheduler_object=self.scheduler_obj,
                simulator_config=self.simulator_config,
                scheduler_config=self.scheduler_config,
                seed=seed,
                verbose=self.verbose,
                save_dir=self.save_dir,
            )
            if not checkpoint:
                print('Simulator failed for config with name {} and seed {}. '
                      'Skipping!'.format(self.scheduler_name, seed))
            else:
                checkpoints.append(checkpoint)

        # Average the results if there are multiple checkpoints
        if not checkpoints:
            print('No available checkpoint objects!')
            return None
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
            avg_mean_regrets = self._average_n_lists(mean_regrets)
            print('Mean regrets', avg_mean_regrets)
            avg_cumulative_regrets = self._average_n_lists(cumulative_regrets)
            print('Cumlative regrets', avg_cumulative_regrets)
            return ExperimentGroupResults(
                self.scheduler_name,
                len(checkpoints),
                avg_mean_regrets,
                avg_cumulative_regrets
            )


if __name__ == '__main__':
    egroup = ExperimentGroup('ASHA', None, [109, 100])
    egroup.run()
