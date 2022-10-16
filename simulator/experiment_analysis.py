# take in 1-N experiment groups
# if > 1, then draw comparison graphs

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from experiment_group import ExperimentGroupResults


# TODO: Combine a lot of functions with ExperimentVisualizer class
class ExperimentAnalysis:

    def _plot_cumulative_regrets(
        cumulative_regrets: List[List[float]],
        cumulative_regret_errors: List[List[float]],
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
        average_regrets: List[np.array],
        average_regrets_error: List[np.array],
        num_workers: int,
        plot_name: str,
        show: bool = False
    ) -> None:
        plt.title('Average Regret for {}-Workers per Epoch'.format(num_workers))
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative per Epoch Instance Regret')
        for average_regret, average_regret_error in zip(average_regrets, average_regrets_error):
            average_regret = np.insert(average_regret, 0, 0)
            average_regret_error = np.insert(average_regret_error, 0, 0)
            plt.plot(list(range(1, len(average_regret)+1)), average_regret)
            plt.fill_between(
                list(range(1, len(average_regret)+1)),
                average_regret + average_regret_error,
                average_regret - average_regret_error,
                alpha=0.4,
                facecolor='blue',
                linewidth=0
            )
        plt.savefig(plot_name)
        if show:
            plt.show()

    def plot_results(results: List[ExperimentGroupResults]) -> None:
        # regrets = [result.avg_cumulative_regret for result in results]
        # errors = [result.avg_cumulative_regret_err for result in results]
        # ExperimentAnalysis._plot_cumulative_regrets(
        #     regrets,
        #     errors,
        #     results[0].num_workers,
        #     'Cumulative Regrets',
        #     show=True
        # )
        regrets = [result.avg_mean_regret for result in results]
        errors = [result.avg_mean_regret_err for result in results]
        ExperimentAnalysis._plot_average_regrets(
            regrets,
            errors,
            results[0].num_workers,
            'Average Regrets',
            show=True
        )

