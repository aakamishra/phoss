import numpy as np
import pandas as pd
from typing import Any, List, Optional, Tuple, Union
from experiment_runner import ExperimentRunner
from itertools import zip_longest
import pdb

DEFAULT_SIMULATOR_CONFIG = "simulator_configs/default_config.json"
DEFAULT_SCHEDULER_CONFIG = "scheduler_configs/default_config.json"
SCHEDULER_CONFIG_NAMES = ["ASHA", "Hyperband", "PBT", "PredASHA", "Median", "Random"]


class ExperimentGroupResults:

    def __init__(
        self,
        scheduler_name: str,
        group_size: int,
        num_workers: int,
        avg_mean_regret: List[float],
        avg_mean_regret_err: List[float],
        avg_cumulative_regret: List[float],
        avg_cumulative_regret_err: List[float],
        moving_loss_avgs: List[float],
        moving_loss_avgs_errs: List[float],
    ):
        self.scheduler_name = scheduler_name
        self.group_size = group_size
        self.num_workers = num_workers
        self.avg_mean_regret = avg_mean_regret
        self.avg_cumulative_regret = avg_cumulative_regret
        self.avg_mean_regret_err = avg_mean_regret_err
        self.avg_cumulative_regret_err = avg_cumulative_regret_err
        self.moving_loss_avgs = moving_loss_avgs
        self.moving_loss_avgs_errs = moving_loss_avgs_errs


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
            avg = (np.sum(values) - (len(values) / self.num_actors)
                  * best_arm_sum) / len(values)
            avgs.append(avg)
            running_sum += avg
            sums.append(running_sum)
        return [avgs, sums]

    @staticmethod
    def _time_processing_column(df, start_time, bin_value=1):
        df["time_interval"] = ((df['timestamp'] - start_time)/(bin_value)).astype(int)
        return df

    def _calculate_average_loss_per_worker(self, data_file: str, bin_value=1) -> List[float]:
        # load in data
        data = data_file = pd.read_csv(data_file)

        # calculate intial start time and bins
        start_time = data.timestamp.min()
        total = data.timestamp.max() - data.timestamp.min()

        # add new time bin column
        joint_logs = self._time_processing_column(data, start_time, bin_value)

        time_log = joint_logs.groupby(['time_interval', 'trial_id']).min()

        # list to save moving average over time
        moving_avg, moving_avg_err = [], []

        # find max time amount
        max_time_interval = max(joint_logs['time_interval'])

        # reset the index to get back the labels for the group by columns
        reset_time_log = time_log.reset_index()
        for i in range(0, max_time_interval+1):
            # for each time interval subset the qualifying configurations
            subset_array = reset_time_log[reset_time_log["time_interval"] <= i]["loss"].values
            # sort the values from smallest to largest for the test loss values
            sorted_subset_array = np.sort(subset_array)
            # get the average of the top 10 values
            top_avgs = np.mean(sorted_subset_array[:self.num_actors])
            top_avgs_err = np.std(sorted_subset_array[:self.num_actors])
            moving_avg.append(top_avgs)
            moving_avg_err.append(top_avgs_err)
        
        return moving_avg, moving_avg_err


    # TODO: Make this a common function
    def _average_n_lists(
        self, lists: List[Union[int, float]],
            override: bool = False
    ) -> Tuple[List[float], List[float]]:
        if not lists:
            return []
        elif len(lists) == 1:
            return lists[0]
        n = len(lists[0])
        for i in range(1, len(lists)):
            if len(lists[i]) != n or override:
                print('Lists must have same size!')
                return []

        vals = np.array(lists)
        avgs = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        return avgs, std

    @staticmethod
    def _average_non_matching_lists(lists):
        l_lens = [len(l) for l in lists]
        max_len = max(l_lens)
        bins = np.zeros(max_len)
        for ln in l_lens:
            bins = bins + np.array([1]*ln + [0]*(max_len - ln))
        lists = [l + [0]*(max_len - len(l)) for l in lists]
        summed = np.sum(lists, axis=0)
        return summed / bins

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
            moving_loss_avgs = []
            moving_loss_avgs_errs = []
            for checkpoint in checkpoints:
                mean_regret, cumulative_regret = self._calculate_regret(
                    checkpoint.true_sim_file, checkpoint.data_file
                )
                moving_avg, moving_avg_err = self._calculate_average_loss_per_worker(checkpoint.data_file)
                moving_loss_avgs.append(moving_avg)
                moving_loss_avgs_errs.append(moving_avg_err)

                mean_regrets.append(mean_regret)
                cumulative_regrets.append(cumulative_regret)

            avg_mean_regrets, avg_mean_regrets_err = self._average_n_lists(mean_regrets)
            # TODO Reconcile which error to use (seed error, or value error)
            moving_loss_avgs = self._average_non_matching_lists(moving_loss_avgs)
            moving_loss_avgs_errs = self._average_non_matching_lists(moving_loss_avgs_errs)
            print('Mean regrets', avg_mean_regrets)
            avg_cumulative_regrets, avg_cumulative_regrets_err = self._average_n_lists(cumulative_regrets)
            print('Cumlative regrets', avg_cumulative_regrets)
            return ExperimentGroupResults(
                self.scheduler_name, #TODO: Give option to pass in different name
                len(checkpoints),
                self.num_actors,
                avg_mean_regrets,
                avg_mean_regrets_err,
                avg_cumulative_regrets,
                avg_cumulative_regrets_err,
                moving_loss_avgs,
                moving_loss_avgs_errs
            )


if __name__ == '__main__':
    egroup = ExperimentGroup('ASHA', None, [109, 100])
    egroup.run()
