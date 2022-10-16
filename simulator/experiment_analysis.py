# take in 1-N experiment groups
# if > 1, then draw comparison graphs

from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from experiment_group import ExperimentGroupResults


# TODO: Combine a lot of functions with ExperimentVisualizer class
class ExperimentAnalysis:
    """Methods to plot and analyze results from an ExperimentGroup"""

    def _plot_cumulative_regrets(
        cumulative_regrets: List[List[float]],
        cumulative_regrets_errors: List[List[float]],
        num_workers: int,
        plot_name: str,
        show: bool = False,
        names: List[str] = [],
    ) -> None:
        ExperimentAnalysis._plot_regrets_margin_of_error(
            cumulative_regrets,
            cumulative_regrets_errors,
            num_workers,
            'cumulative',
            plot_name,
            names=names,
            show=show
        )

    def _plot_average_regrets(
        average_regrets: List[np.array],
        average_regrets_error: List[np.array],
        num_workers: int,
        plot_name: str,
        show: bool = False,
        names: List[str] = [],
    ) -> None:
        ExperimentAnalysis._plot_regrets_margin_of_error(
            average_regrets,
            average_regrets_error,
            num_workers,
            'average',
            plot_name,
            names=names,
            show=show
        )

    def _plot_regrets_margin_of_error(
        regrets: List[np.array],
        errors: List[np.array],
        num_workers: int,
        regret_type: str,
        save_as: str,
        names: List[str] = [],
        show: bool = False
    ) -> None:
        if not regrets or not errors:
            print('Regret and error matrices must be non-empty!')
            return
        if not names:
            names = ['' for _ in regrets[0]]

        plt.title(
            '{} Regret for {} Wokers Per Epoch'.format(
                regret_type.capitalize(), num_workers)
        )
        plt.xlabel('Epochs')
        plt.ylabel(
            '{} per Epoch Instance Regret'.format(regret_type.capitalize())
        )
        for regret, error, name in zip(regrets, errors, names):
            regret = np.insert(regret, 0, 0)
            error = np.insert(error, 0, 0)
            plt.plot(list(range(1, len(regret)+1)), regret,
                     label=name if name else None)
            plt.fill_between(
                list(range(1, len(regret)+1)),
                regret + error,
                regret - error,
                alpha=0.4,
                facecolor='blue',
                linewidth=0
            )
        plt.legend()
        plt.savefig(save_as)
        if show:
            plt.show()

    def plot_results(results: List[ExperimentGroupResults]) -> None:
        regrets = [result.avg_cumulative_regret for result in results]
        errors = [result.avg_cumulative_regret_err for result in results]
        names = [result.scheduler_name for result in results]
        ExperimentAnalysis._plot_cumulative_regrets(
            regrets,
            errors,
            results[0].num_workers,
            'Cumulative Regrets',
            show=True,
            names=names,
        )
        regrets = [result.avg_mean_regret for result in results]
        errors = [result.avg_mean_regret_err for result in results]
        ExperimentAnalysis._plot_average_regrets(
            regrets,
            errors,
            results[0].num_workers,
            'Average Regrets',
            show=True,
            names=names,
        )

