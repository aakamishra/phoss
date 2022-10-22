from phoss.experiment_visualizer import ExperimentVisualizer

checkpoint_file = '/Users/amishra/DHPOSS/DHPOSS/simulator/test/checkpoint-2022-10-13-09-50-21.json'
viz = ExperimentVisualizer(checkpoint_file)
viz.plot_loss_curves('checkpoint-2022-10-13-09-50-21.png')
viz.get_heatmap('checkpoint-2022-10-13-09-50-21-heatmap.png')
