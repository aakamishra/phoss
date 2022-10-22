from typing import List
import matplotlib.pyplot as plt
import numpy as np
from phoss.experiment_group import ExperimentGroupResults


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
        plot_colors = ['lawngreen', 'cyan', 'magenta', 'salmon']
        error_colors = ['yellow', 'blue', 'purple', 'red']
        inds = list(range(len(regrets)))
        for regret, error, name, i in zip(regrets, errors, names, inds):
            regret = np.insert(regret, 0, 0)
            error = np.insert(error, 0, 0)
            plt.plot(list(range(1, len(regret)+1)), regret,
                     label=name if name else None, color=plot_colors[i])
            plt.fill_between(
                list(range(1, len(regret)+1)),
                regret + error,
                regret - error,
                alpha=0.2,
                facecolor=error_colors[i],
                linewidth=0
            )
        plt.legend()
        plt.savefig(save_as)
        if show:
            plt.show()
        plt.close()

    def _plot_average_moving_loss(
        average_moving_loss: List[np.array],
        average_moving_loss_error: List[np.array],
        num_workers: int,
        plot_name: str,
        show: bool = False,
        names: List[str] = [],
    ) -> None:
        plt.title('Top {} Averaged Moving Loss over Time'.format(num_workers))
        plt.xlabel('Time Bins')
        plt.ylabel('Averaged Loss')
        for average, error, name in zip(average_moving_loss, average_moving_loss_error, names):
            plt.plot(list(range(1, len(average)+1)), average, label=name)
            plt.fill_between(
                list(range(1, len(average)+1)),
                average + error,
                average - error,
                alpha=0.2,
                facecolor='blue',
                linewidth=0
            )
        plt.legend()
        plt.savefig(plot_name)
        if show:
            plt.show()
        plt.close()

    def plot_results(
        results: List[ExperimentGroupResults], show: bool = True
    ) -> None:
        regrets = [result.avg_cumulative_regret for result in results]
        errors = [result.avg_cumulative_regret_err for result in results]
        names = [result.scheduler_name for result in results]
        ExperimentAnalysis._plot_cumulative_regrets(
            regrets,
            errors,
            results[0].num_workers,
            'Cumulative_Regrets',
            show=show,
            names=names,
        )

        regrets = [result.avg_mean_regret for result in results]
        errors = [result.avg_mean_regret_err for result in results]
        ExperimentAnalysis._plot_average_regrets(
            regrets,
            errors,
            results[0].num_workers,
            'Average_Regrets',
            show=show,
            names=names,
        )

        averages = [result.moving_loss_avgs for result in results]
        errors = [result.moving_loss_avgs_errs for result in results]
        ExperimentAnalysis._plot_average_moving_loss(
            averages,
            errors,
            results[0].num_workers,
            'Average_Moving_Loss',
            show=show,
            names=names,
        )

