from collections import OrderedDict
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ExperimentVisualizer:

    def __init__(
        self,
        checkpoint_file: str = None,
        checkpoint_dict=None,
        show=False
    ):
        if checkpoint_dict:
            config = checkpoint_dict
            self.num_workers = config.get('num_actors', 8)
            self.max_num_epochs = config.get('max_num_epochs', 100)
            self.scheduler_name = config.get('scheduler_name', '')
            self.gen_sim_file = config.get('gen_sim_file', '')
            self.true_sim_file = config.get('true_sim_file', '')
            self.simulation_name = config.get('simulation_name', '')
            self.data_file = config.get('data_file', '')
            self.num_samples = config.get('num_samples', 0)

        else:
            with open(checkpoint_file, encoding='utf-8') as f:
                print(checkpoint_file)
                config = json.load(f)

                self.num_workers = config.get('num_actors', 8)
                self.max_num_epochs = config.get('max_num_epochs', 100)
                self.scheduler_name = config.get('scheduler_name', '')
                self.gen_sim_file = config.get('gen_sim_file', '')
                self.true_sim_file = config.get('true_sim_file', '')
                self.simulation_name = config.get('simulation_name', '')
                self.data_file = config.get('data_file', '')
                self.num_samples = config.get('num_samples', '')

        self.simulated_loss = np.genfromtxt(self.gen_sim_file, delimiter=',')
        self.true_loss = np.genfromtxt(self.true_sim_file, delimiter=',')
        self.total_data_file = pd.read_csv(self.data_file)
        self.average_regret = []
        self.cumulative_regret = []
        self.show = show

    def get_heatmap(self, file_name: str):
        """Generate Scheduler Loss Heat Map"""
        plt.figure(figsize=(8, 8))
        sns.heatmap(self.simulated_loss)

        totals = self.total_data_file.groupby(
            'index'
        )['training_iteration'].max().values
        for i in range(len(totals)):
            plt.axvline(i + 0.5, 1, 1 - totals[i] / 100, linewidth=0.9,
                        color='white')

        plt.xlabel('Samples')
        plt.ylabel('Epochs')
        plt.title('Scheduled Runs Over Loss Surface')
        plt.savefig(file_name)
        if self.show:
            plt.show()

    def plot_loss_curves(self, plot_name: str):
        time_range = list(range(self.max_num_epochs))
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
        plt.plot(time_range, self.simulated_loss, alpha=0.1, color='blue',
                 label='Simulated Loss')
        plt.plot(time_range, self.true_loss, alpha=0.15, color='red',
                 label='True Mean Schedule')
        best_path = self.total_data_file.groupby(
            'training_iteration'
        )['loss'].min().values
        plt.plot(
            time_range,
            best_path,
            label='Scheduler Chosen Path Best Value',
            color='cyan'
        )
        best_average_worker_path = self.total_data_file.groupby(
            'training_iteration'
        )['loss'].apply(lambda x: x.nsmallest(self.num_workers).mean())
        plt.plot(
            time_range,
            best_average_worker_path,
            label=f'Scheduler Chosen Average {self.num_workers}-Worker Path Value',
            color='lime'
        )
        ax.set_facecolor('silver')
        # Hide the right and top spines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.title('Simulation Loss Curve Path Overview')
        plt.ylabel('Simulated Loss Values')
        plt.xlabel('Epochs')
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.grid()
        plt.savefig(plot_name)
        if self.show():
            plt.show()

    def calculate_regret(self):
        """Get average and cumulative regret"""
        avgs = []
        sums = []
        running_sum = 0

        for i in range(1, self.max_num_epochs):
            indices = self.total_data_file[self.total_data_file['training_iteration'] == i]['index'].unique()
            values = self.true_loss[i, indices]
            best_arm_sum = np.sum(sorted(self.true_loss[i])[:self.num_workers])
            avg = (np.sum(values) - (len(values) / self.num_workers)
                   * best_arm_sum) / len(values)
            avgs.append(avg)
            running_sum += avg
            sums.append(running_sum)
        self.average_regret = avgs
        self.cumulative_regret = sums

    def plot_cumulative_regret(self, plot_name: str):
        plt.title(f'Average Regret for {self.num_workers}-Workers per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        plt.plot(list(range(1, self.max_num_epochs)), self.cumulative_regret)
        plt.savefig(plot_name)
        if self.show:
            plt.show()

    def plot_average_regret(self, plot_name: str):
        plt.title(f'Average Regret for {self.num_workers}-Workers per Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        plt.plot(list(range(1, self.max_num_epochs)), self.average_regret)
        plt.savefig(plot_name)
        if self.show():
            plt.show()

    def combined_seed_data_regret_curve(self, experiment_list, plot_name: str):
        cumulative_avg = self.average_regret
        for experiment in experiment_list:
            cumulative_avg = np.vstack(
                [cumulative_avg, experiment.average_regret]
            )
        x = list(range(1, self.max_num_epochs))
        y = np.mean(cumulative_avg, axis=0)
        error = np.std(cumulative_avg, axis=0)
        plt.plot(x, y, 'k')
        plt.fill_between(x, y - error, y + error, alpha=0.6, facecolor='blue',
                         linewidth=0)
        plt.title('Averaged Regret Over Various Seeds')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(plot_name)
        if self.show():
            plt.show()
