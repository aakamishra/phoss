from experiment_visualizer import ExperimentVisualizer
from ray_runner import RayRunnerAPI


seeds = [184, 243, 161, 109, 165]
check_dicts = []
viz_list = []
for seed in seeds:
    checkpoint_dict = RayRunnerAPI.call_simulator(
                'ASHA',
                num_samples=100,
                max_num_epochs=100,
                gpus_per_trial=0,
                cpus_per_trial=1,
                num_actors=16,
                seed=seed,
                verbose=1,
                save_dir="test")
    check_dicts.append(checkpoint_dict)

    viz = ExperimentVisualizer(checkpoint_dict=checkpoint_dict)
    viz_list.append(viz)
    viz.plot_loss_curves(f"{viz.simulation_name}-loss-curve.png")
    viz.get_heatmap(f"{viz.simulation_name}-heatmap.png")
    viz.plot_average_regret(f"{viz.simulation_name}-average-regret.png")
    viz.plot_cumulative_regret(f"{viz.simulation_name}-cumulative-regret.png")
