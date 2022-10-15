from typing import Optional
from schedulers import Scheduler
from trainables import SimulatedTrainable
from landscaper import NormalLossDecayLandscape
from ray import tune
from ray.tune.logger import CSVLoggerCallback
import os
from ray.air.config import RunConfig
import ray
import argparse
from datetime import datetime
import json
import pandas as pd
import numpy as np
from ray_runner import RayRunner


DEFAULT_SIMULATOR_CONFIG = "simulator_configs/default_config.json"
DEFAULT_SCHEDULER_CONFIG = "scheduler_configs/default_config.json"
SCHEDULER_CONFIG_NAMES = ["ASHA", "Hyperband", "PBT", "PredASHA"]


class CheckpointObject:

    def __init__(
        self,
        num_actors: int,
        max_num_epochs: int,
        scheduler_name: str,
        gen_sim_file: str,
        true_sim_file: str,
        simulation_name: str,
        data_file: str, num_samples: int
    ):
        self.num_actors = num_actors
        self.max_num_epochs = max_num_epochs
        self.scheduler_name = scheduler_name
        self.gen_sim_file = gen_sim_file
        self.true_sim_file = true_sim_file
        self.simulation_name = simulation_name
        self.data_file = data_file
        self.num_samples = num_samples

    def persist_json(self, filename: str):
        with open(filename, 'w') as fp:
            json.dump(
                {
                    "num_actors": self.num_actors,
                    "max_num_epochs": self.max_num_epochs,
                    "scheduler_name": self.scheduler_name,
                    "gen_sim_file": self.gen_sim_file,
                    "true_sim_file": self.true_sim_file,
                    "simulation_name": self.simulation_name,
                    "data_file": self.data_file,
                    "num_samples": self.num_samples,
                },
                fp,
                indent=4
            )


class ExperimentRunner:

    def call_simulator(
        sched_name: str,
        num_samples: int = 16,
        max_num_epochs: int = 10,
        gpus_per_trial: int = 0,
        cpus_per_trial: int = 1,
        num_actors: int = 4,
        seed: int = 109,
        scheduler_object = None,
        simulator_config: str = DEFAULT_SIMULATOR_CONFIG,
        scheduler_config: str = DEFAULT_SCHEDULER_CONFIG,
        verbose: int = 0,
        save_dir: str = '',
    ) -> Optional[CheckpointObject]:
        """
        Public function to be called as an API endpoint.
        """
        if sched_name.lower() == "custom":
            if not scheduler_object:
                print("Custom scheduler object not provided!")
                return None
            return ExperimentRunner._call_custom_simulator(
                scheduler_object,
                num_samples=num_samples,
                max_num_epochs=max_num_epochs,
                gpus_per_trial=gpus_per_trial,
                cpus_per_trial=cpus_per_trial,
                num_actors=num_actors,
                seed=seed,
                verbose=verbose,
                save_dir=save_dir)
        else:
            if sched_name not in SCHEDULER_CONFIG_NAMES:
                print("Could not find sched_name {} in \
                    SCHEDULER_CONFIG_NAMES".format(sched_name))
                return None
            return ExperimentRunner._call_common_simulator(
                sched_name,
                num_samples=num_samples,
                max_num_epochs=max_num_epochs,
                gpus_per_trial=gpus_per_trial,
                cpus_per_trial=cpus_per_trial,
                num_actors=num_actors,
                seed=seed,
                simulator_config=simulator_config,
                scheduler_config=scheduler_config,
                verbose=verbose,
                save_dir=save_dir)

    def _call_common_simulator(
        sched_name: str,
        num_samples: int = 16,
        max_num_epochs: int = 10,
        gpus_per_trial: int = 0,
        cpus_per_trial: int = 1,
        num_actors: int = 4,
        seed: int = 109,
        simulator_config: str = DEFAULT_SIMULATOR_CONFIG,
        scheduler_config: str = DEFAULT_SCHEDULER_CONFIG,
        verbose: int = 0,
        save_dir: str = '',
    ) -> CheckpointObject:
        """
        Helper method used to call RayRunner on a common scheduler such as ASHA,
        Hyperband, or PTB.
        To be called from `ExperimentRunner.call_simulator`.
        """
        # loading scheduler config
        if verbose:
            print("Loading config file for scheduler: ", scheduler_config)
        with open(scheduler_config, encoding='utf-8') as f:
            scheduler_config = json.load(f)
        scheduler_config['max_t'] = max_num_epochs

        if verbose:
            print("Initializing Ray Runner")
        runner = RayRunner(
            num_samples=num_samples,
            num_actors=num_actors,
            cpus_per_trial=cpus_per_trial,
            gpus_per_trial=gpus_per_trial,
            simulator_config=simulator_config,
            scheduler_config=scheduler_config,
            max_num_epochs=max_num_epochs,
            scheduler_name=sched_name,
            seed=seed)
        return ExperimentRunner._run_simulation(runner, verbose=verbose,
                                            save_dir=save_dir)

    def _call_custom_simulator(
        scheduler,
        num_samples: int = 16,
        max_num_epochs: int = 10,
        gpus_per_trial: int = 0,
        cpus_per_trial: int = 1,
        num_actors: int = 4,
        seed: int = 109,
        verbose: int = 0,
        save_dir: str = '',
    ) -> CheckpointObject:
        """
        Helper method used to call Ray Runner on a custom-defined scheduler.
        To be called from `ExperimentRunner.call_simulator`.
        """
        # loading scheduler config
        if verbose:
            print("Loading config file for scheduler: ", scheduler_config)
        with open(scheduler_config, encoding='utf-8') as f:
            scheduler_config = json.load(f)
        scheduler_config['max_t'] = max_num_epochs

        if verbose:
            print("Initializing Ray Runner")
        runner = RayRunner(
            num_samples=num_samples,
            num_actors=num_actors,
            cpus_per_trial=cpus_per_trial,
            gpus_per_trial=gpus_per_trial,
            scheduler_object=scheduler,
            max_num_epochs=max_num_epochs,
            scheduler_name="custom",
            seed=seed)
        return ExperimentRunner._run_simulation(runner, verbose=verbose,
                                            save_dir=save_dir)

    def _run_simulation(
        runner: RayRunner, verbose: bool = 0, save_dir: str = ''
    ) -> CheckpointObject:
        """
        Runs the simulator given a RayRunner instance and saves the results as a
        set of CSV files.
        """
        if verbose: print("Generating loss simulation")
        runner.generate_simulation()

        if verbose: print("Running Ray Tune Program")
        timestamp = datetime.now()
        results = runner.run()

        if verbose: print("Moving data to checkpoint csv")
        dfs = {result.log_dir: result.metrics_dataframe for result in results}
        data = pd.concat(dfs.values(), ignore_index=True)

        path = os.path.join(os.getcwd(), save_dir) if save_dir else os.getcwd()
        print("Saving results at", path)
        if not os.path.exists(path):
            os.mkdir(path)
        true_sim_path = os.path.join(path,
                                     runner.simulation_name + "-true-sim.csv")
        gen_sim_path = os.path.join(path,
                                    runner.simulation_name + "-gen-sim.csv")
        np.savetxt(true_sim_path, runner.landscaper.true_loss, delimiter=",")
        np.savetxt(gen_sim_path, runner.landscaper.simulated_loss, delimiter=",")

        # move total data to csv
        data_path = os.path.join(path, runner.simulation_name + "-data.csv")
        data.to_csv(data_path)

        # perform checkpointing
        checkpoint = CheckpointObject(
            runner.num_actors,
            runner.max_num_epochs,
            runner.scheduler_name,
            gen_sim_path,
            true_sim_path,
            runner.simulation_name,
            data_path,
            runner.num_samples)
        serialized_timestamp = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
        fcheckpoint = os.path.join(
            path, "checkpoint-{}.json".format(serialized_timestamp))
        checkpoint.persist_json(fcheckpoint)

        print("Finished")
        return checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sched_name', type=str, default="ASHA")
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--max-num-epochs', type=int, default=100)
    parser.add_argument('--gpus-per-trial', type=int, default=0)
    parser.add_argument('--cpus-per-trial', type=int, default=1)
    parser.add_argument('--num-actors', type=int, default=8)
    parser.add_argument('--seed', type=int, default=109)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--simulator-config',
                        type=str,
                        default=DEFAULT_SIMULATOR_CONFIG)
    parser.add_argument('--scheduler-config',
                        type=str,
                        default=DEFAULT_SCHEDULER_CONFIG)
    parser.add_argument('--save', type=str, default='')

    args = parser.parse_args()

    try:
        if args.verbose:
            print("Starting main program...")
        ExperimentRunner.call_simulator(
            args.sched_name,
            num_samples=args.num_samples,
            max_num_epochs=args.max_num_epochs,
            gpus_per_trial=args.gpus_per_trial,
            cpus_per_trial=args.cpus_per_trial,
            num_actors=args.num_actors,
            simulator_config=args.simulator_config,
            scheduler_config=args.scheduler_config,
            seed=args.seed,
            verbose=args.verbose,
            save_dir=args.save)
    except Exception as e:
        print(e)
