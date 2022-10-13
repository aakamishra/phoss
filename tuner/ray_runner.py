from simulators.landscaper import NormalLossDecayLandscape
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.tune.logger import CSVLoggerCallback
import os
from ray.air.config import RunConfig
import random
import ray




def train_simulator(config):
    values = config['sample_array']
    series = values[:,config['index']]
    n = len(series)

    for epoch in range(n):
        loss = series[epoch]
        session.report({'loss': loss})


def main(sched_name, num_samples: int = 16, max_num_epochs: int = 10,
         gpus_per_trial: int = 4/4, cpus_per_trial: int = 1, num_actors=1) -> None:

    algo = None

    # generate loss landscape
    landscaper = NormalLossDecayLandscape(
        max_time_steps=max_num_epochs, samples=num_samples)
    sim_loss = landscaper.generate_landscape()

    config = {}
    config['sample_array'] = sim_loss
    config['index'] = tune.grid_search(list(range(num_samples)))
    
    src_files = ['raytune_engine.py']
    formatted_src_files = [
        os.getcwd() + '/' + file for file in src_files
    ]
    runtime_env = {'includes': formatted_src_files}
    ray.init(runtime_env=runtime_env, 
        include_dashboard=False,
        ignore_reinit_error=True,
        _system_config={"num_heartbeats_timeout": 800, "object_timeout_milliseconds":9000000})

    scheduler = ASHAScheduler(max_t=max_num_epochs, brackets=3,
                         grace_period=1,
                         reduction_factor=4)

    tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(train_simulator),
                resources={'cpu': cpus_per_trial, 'gpu': gpus_per_trial}
            ),
            tune_config=tune.TuneConfig(
                metric='loss',
                mode='min',
                scheduler=scheduler,
                search_alg=algo,
                num_samples=num_samples
            ),
            param_space=config,
            run_config=RunConfig(
                local_dir=os.getcwd(),
                name=f"test-run-s-{num_samples}-t-{max_num_epochs}-{sched_name}",
                callbacks=[CSVLoggerCallback()],
                verbose=1,
                sync_config=tune.SyncConfig(
                syncer=None  # Disable syncing
                )
            ),  
        )
    results = tuner.fit()

    # Obtain a trial dataframe from all run trials of this `tune.run` call.
    dfs = {result.log_dir: result.metrics_dataframe for result in results}
