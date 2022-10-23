import os
import ray
from ray import tune
from ray.air.config import RunConfig
from ray.tune.logger import CSVLoggerCallback
import phoss.common
from phoss.landscaper import NormalLossDecayLandscape
from phoss.schedulers import Scheduler
from phoss.trainables import SimulatedTrainable


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
        simulator_config: str = phoss.common.DEFAULT_SIMULATOR_CONFIG,
        scheduler_config: dict = None,
        verbose: int = 0
    ):

        # search and scheduler objects
        self.scheduler_name = scheduler_name
        if scheduler_name.lower() == 'custom':
            assert scheduler_object is not None
            self.scheduler = scheduler_object
        else:
            assert scheduler_name in phoss.common.SCHEDULER_CONFIG_NAMES
            self.scheduler = Scheduler(
                scheduler_name,
                scheduler_config
            ).get_instance()

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
        self.gen_sim_path = ''

        # randomness
        self.seed = seed

        # simulation name and logging information
        self.simulation_name = 'test-run-n-{}-t-{}-{}-seed-{}'.format(
            self.num_samples, self.max_num_epochs, self.scheduler_name, seed)
        self.verbose = verbose

    def generate_simulation(self) -> True:
        """Function for generated the loss landscape for training on"""

        # create normal landscape
        self.landscaper = NormalLossDecayLandscape(
            seed=self.seed,
            max_time_steps=self.max_num_epochs,
            samples=self.num_samples,
            json_config=self.simulator_config,
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
        ray.init(runtime_env=runtime_env, include_dashboard=False,
                 ignore_reinit_error=True,
                 _system_config={'num_heartbeats_timeout': 800,
                                 'object_timeout_milliseconds': 9000000})


        print('Passing path:',  self.gen_sim_path)
        search_config = {
            'gen_sim_path': self.gen_sim_path,
            'index': tune.grid_search(list(range(self.num_samples))),
        }

        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(SimulatedTrainable),
                resources={'cpu': self.cpus, 'gpu': self.gpus},
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
                ),
            ),
        )

        results = tuner.fit()
        return results
