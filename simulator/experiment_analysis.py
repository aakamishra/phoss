# take in 1-N experiment groups
# if > 1, then draw comparison graphs

import matplotlib.pyplot as plt
from typing import List
from experiment_group import ExperimentGroupResults


# TODO: Combine a lot of functions with ExperimentVisualizer class
class ExperimentAnalysis:

    def _plot_cumulative_regret(cumulative_regret: List[float],
                                num_workers: int, plot_name: str,
                                show: bool = False):
        plt.title('Cumulative Regret for {}-Workers per Epoch'.format(num_workers))
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        print(cumulative_regret)
        cumulative_regret.insert(0, 0)
        print(cumulative_regret)
        plt.plot(list(range(1, len(cumulative_regret)+1)), cumulative_regret)
        plt.savefig(plot_name)
        if show:
            plt.show()

    def _plot_average_regret(average_regret: List[float], num_workers: int,
                             plot_name: str, show: bool = False):
        plt.title('Average Regret for {}-Workers per Epoch'.format(num_workers))
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        average_regret.insert(0, 0)
        plt.plot(list(range(1, len(average_regret)+1)), average_regret)
        plt.savefig(plot_name)
        if show:
            plt.show()

    def plot_results(results: List[ExperimentGroupResults]) -> None:
        if len(results) == 1:
            ExperimentAnalysis._plot_cumulative_regret(
                results[0].avg_cumulative_regret,
                results[0].num_workers,
                'Name',
                show=True
            )
            ExperimentAnalysis._plot_average_regret(
                results[0].avg_mean_regret,
                results[0].num_workers,
                'Name',
                show=True
            )

