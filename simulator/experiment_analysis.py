# take in 1-N experiment groups
# if > 1, then draw comparison graphs

import matplotlib.pyplot as plt
from typing import List
from experiment_group import ExperimentGroupResults


# TODO: Combine a lot of functions with ExperimentVisualizer class
class ExperimentAnalysis:

    def _plot_cumulative_regrets(
        cumulative_regrets: List[List[float]],
        num_workers: int,
        plot_name: str,
        show: bool = False
    ) -> None:
        plt.title('Cumulative Regret for {}-Workers per Epoch'.format(num_workers))
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        for cumulative_regret in cumulative_regrets:
            cumulative_regret.insert(0, 0)
            plt.plot(list(range(1, len(cumulative_regret)+1)), cumulative_regret)
        plt.savefig(plot_name)
        if show:
            plt.show()

    def _plot_average_regrets(
        average_regrets: List[List[float]],
        num_workers: int,
        plot_name: str,
        show: bool = False
    ) -> None:
        plt.title('Average Regret for {}-Workers per Epoch'.format(num_workers))
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        for average_regret in average_regrets:
            average_regret.insert(0, 0)
            plt.plot(list(range(1, len(average_regret)+1)), average_regret)
        plt.savefig(plot_name)
        if show:
            plt.show()

    def plot_results(results: List[ExperimentGroupResults]) -> None:
        regrets = [result.avg_cumulative_regret for result in results]
        ExperimentAnalysis._plot_cumulative_regrets(
            regrets,
            results[0].num_workers,
            'Cumulative Regrets',
            show=True
        )
        regrets = [result.avg_mean_regret for result in results]
        ExperimentAnalysis._plot_average_regrets(
            regrets,
            results[0].num_workers,
            'Average Regrets',
            show=True
        )

