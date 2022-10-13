from ray_runner import RayRunnerAPI

RayRunnerAPI.call_simulator(
            'ASHA',
            num_samples=100,
            max_num_epochs=100,
            gpus_per_trial=0,
            cpus_per_trial=1,
            num_actors=8,
            seed=143,
            verbose=1,
            save_dir="test")

