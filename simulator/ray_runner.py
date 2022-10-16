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


DEFAULT_SIMULATOR_CONFIG = 'simulator_configs/default_config.json'
DEFAULT_SCHEDULER_CONFIG = 'scheduler_configs/default_config.json'
SCHEDULER_CONFIG_NAMES = ['ASHA', 'Hyperband', 'PBT', 'PredASHA']


class RayRunner:
    def __init__(
        self,
        scheduler_name: str = 'ASHA',
        num_samples: int = 16,
        max_num_epochs: int = 10,
        gpus_per_trial: int = 1,
        cpus_per_trial: int = 1,
        num_actors: int = 1,
        seed: int = 109,
        algo=None,
        scheduler_object=None,
        simulator_config: str = None,
        scheduler_config: dict = None,
        verbose: int = 0
    ):

        # search and scheduler objects
        self.scheduler_name = scheduler_name
        if scheduler_name.lower() == 'custom':
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
        self.simulation_name = 'test-run-n-{}-t-{}-{}-seed-{}'.format(
            self.num_samples, self.max_num_epochs, self.scheduler_name, seed)
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
            starting_mu_args=simulator_config_dict['STARTING_MU_ARGS'],
            ending_mu_args=simulator_config_dict['ENDING_MU_ARGS'],
            starting_std_args=simulator_config_dict['STARTING_STD_ARGS'],
            ending_std_args=simulator_config_dict['ENDING_STD_ARGS']
        )

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
                 _system_config={'num_heartbeats_timeout': 800,
                                 'object_timeout_milliseconds': 9000000})

        search_config = {
            'sample_array': self.landscaper.simulated_loss,
            'index': tune.grid_search(
                list(
                    range(
                        self.num_samples)))}

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
                verbose=self.verbose,
                sync_config=tune.SyncConfig(
                    syncer=None  # Disable syncing
                )
            ),
        )

        results = tuner.fit()
        return results
