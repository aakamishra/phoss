from schedulers import Scheduler
from trainables import SimulatedTrainable
from landscaper import NormalLossDecayLandscape
from ray import tune
from ray.tune.logger import CSVLoggerCallback
import os
from ray.air.config import RunConfig
import ray
import argparse
import json
import pandas as pd
import numpy as np


DEFAULT_SIMULATOR_CONFIG = "simulator_configs/default_config.json"
DEFAULT_SCHEDULER_CONFIG = "scheduler_configs/default_config.json"
SCHEDULER_CONFIG_NAMES = ["ASHA", "Hyperband", "PBT", "PredASHA"]


class RayRunner:
    def __init__(self,
                 scheduler_name: str = "ASHA",
                 num_samples: int = 16,
                 max_num_epochs: int = 10,
                 gpus_per_trial: int = 1,
                 cpus_per_trial: int = 1,
                 num_actors: int = 1,
                 seed: int = 109,
                 algo = None,
                 scheduler_object = None,
                 simulator_config: str = None,
                 scheduler_config: dict = None,
                 verbose: int = 0):

        # search and scheduler objects
        self.scheduler_name = scheduler_name
        if scheduler_name.lower() == "custom":
            assert scheduler_object is not None
            self.scheduler = scheduler_object
        else:
            self.scheduler = Scheduler(
                scheduler_name, scheduler_config).get_instance()

        self.algo = algo
        self.simulator_config = simulator_config
        self.landscaper = None

        # hardware specifications
        self.gpus = gpus_per_trial
        self.cpus = cpus_per_trial

        # simulation dimension length arguments
        self.max_num_epochs = max_num_epochs
        self.num_actors = num_actors
        self.num_samples = num_samples

        # randomness
        self.seed = seed

        # simulation name and logging information
        self.simulation_name = f"test-run-n-{self.num_samples}-t-{self.max_num_epochs}-{self.scheduler_name}-seed-{seed}"
        self.verbose = verbose

    def generate_simulation(self) -> True:
        """Function for generated the loss landscape for training on"""

        # load the simulator configuration files enumerating the distribution
        simulator_config_dict = {}
        if self.simulator_config:
            with open(self.simulator_config, encoding='utf-8') as f:
                simulator_config_dict = json.load(f)
        else:
            with open(DEFAULT_SIMULATOR_CONFIG, encoding='utf-8') as f:
                simulator_config_dict = json.load(f)

        # create normal landscape
        self.landscaper = NormalLossDecayLandscape(
            seed=self.seed,
            max_time_steps=self.max_num_epochs,
            samples=self.num_samples,
            starting_mu_args=simulator_config_dict["STARTING_MU_ARGS"],
            ending_mu_args=simulator_config_dict["ENDING_MU_ARGS"],
            starting_std_args=simulator_config_dict["STARTING_STD_ARGS"],
            ending_std_args=simulator_config_dict["ENDING_STD_ARGS"])

        self.landscaper.generate_landscape()

    def run(self):
        """Function for running the main tuner pipeline"""

        # define files to be used for the runtime environment
        src_files = ['ray_runner.py']
        formatted_src_files = [
            os.getcwd() + '/' + file for file in src_files
        ]
        runtime_env = {'includes': formatted_src_files}

        # initialize Ray RPC server
        ray.init(runtime_env=runtime_env,
                 include_dashboard=False,
                 ignore_reinit_error=True,
                 _system_config={"num_heartbeats_timeout": 800,
                                 "object_timeout_milliseconds": 9000000})


        search_config = {'sample_array': self.landscaper.simulated_loss,
                         'index': tune.grid_search(list(range(self.num_samples)))
                         }

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(SimulatedTrainable),
                resources={'cpu': self.cpus, 'gpu': self.gpus}
            ),
            tune_config=tune.TuneConfig(
                metric='loss',
                mode='min',
                scheduler=self.scheduler,
                num_samples=1,
                max_concurrent_trials=self.num_actors,
                reuse_actors=True,
            ),
            param_space=search_config,
            run_config=RunConfig(
                local_dir=os.getcwd() + '/results',
                name=self.simulation_name,
                callbacks=[CSVLoggerCallback()],
                verbose=0,
                sync_config=tune.SyncConfig(
                    syncer=None  # Disable syncing
                )
            ),
        )

        results = tuner.fit()
        ray.shutdown()

        return results


class RayRunnerAPI:

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
    ) -> None:
        """
        Public function to be called as an API endpoint.
        """
        if sched_name.lower() == "custom":
            if not scheduler_object:
                print("Custom scheduler object not provided!")
                return
            RayRunnerAPI._call_custom_simulator(
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
            RayRunnerAPI._call_common_simulator(
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
    ) -> None:
        """
        Helper method used to call RayRunner on a common scheduler such as ASHA,
        Hyperband, or PTB.
        To be called from `RayRunnerAPI.call_simulator`.
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
        RayRunnerAPI._run_simulation(runner, verbose=verbose, save_dir=save_dir)

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
    ) -> None:
        """
        Helper method used to call Ray Runner on a custom-defined scheduler.
        To be called from `RayRunnerAPI.call_simulator`.
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
        RayRunnerAPI._run_simulation(runner, verbose=verbose, save_dir=save_dir)

    def _run_simulation(
        runner: RayRunner, verbose: bool = 0, save_dir: str = ''
    ) -> None:
        """
        Runs the simulator given a RayRunner instance and saves the results as a
        set of CSV files.
        """
        if verbose: print("Generating loss simulation")
        runner.generate_simulation()

        if verbose: print("Running Ray Tune Program")
        results = runner.run()

        if verbose: print("Moving data to checkpoint csv")
        dfs = {result.log_dir: result.metrics_dataframe for result in results}
        data = pd.concat(dfs.values(), ignore_index=True)

        path = os.path.join(os.getcwd(), save_dir) if save_dir else os.getcwd()
        print("Saving results at", path)
        if not os.path.exists(path):
            os.mkdir(path)
        np.savetxt(os.path.join(path, runner.simulation_name + "-true-sim.csv"),
                   runner.landscaper.true_loss, delimiter=",")
        np.savetxt(os.path.join(path, runner.simulation_name + "-gen-sim.csv"),
                   runner.landscaper.simulated_loss, delimiter=",")

        # move total data to csv
        data.to_csv(os.path.join(path, runner.simulation_name + "-data.csv"))

        print("done.")


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
        if args.verbose: print("Starting main program...")
        RayRunnerAPI.call_simulator(
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
