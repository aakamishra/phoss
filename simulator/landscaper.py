import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable
import matplotlib
matplotlib.use('TkAgg')

DEFAULT_STARTING_MU_ARGS = {"mu": 1.0, "std": 0.1}
DEFAULT_ENDING_MU_ARGS = {"mu": 0.3, "std": 0.05}
DEFAULT_STARTING_STD_ARGS = {"mu": 0.02, "std": 0.005}
DEFAULT_ENDING_STD_ARGS = {"mu": 0.001, "std": 0.005}


class NormalLossDecayLandscape:
    def __init__(self, 
                max_time_steps: int = 100,
                samples: int = 50,
                starting_mu_args: dict = DEFAULT_STARTING_MU_ARGS,
                starting_mu_dist: Callable = np.random.normal,
                ending_mu_args: dict = DEFAULT_ENDING_MU_ARGS,
                ending_mu_dist: Callable = np.random.normal,
                starting_std_args: dict = DEFAULT_STARTING_STD_ARGS,
                starting_std_dist: Callable = np.random.normal,
                ending_std_args: dict = DEFAULT_ENDING_STD_ARGS,
                ending_std_dist: Callable = np.random.normal,
                noise_dist: Callable = np.random.normal,
                seed: int = 109,
                absolute: bool = True,
                loss_curve_shape: str = "expo",
                std_curve_shape: str = "linear",):

        # set random seed for experiment replicability
        self.seed = seed
        np.random.seed(self.seed)

        # store instance values
        self.absolute = absolute
        self.loss_curve_shape = loss_curve_shape
        self.samples = samples
        self.max_time_steps = max_time_steps
        self.noise_dist = noise_dist
        self.simulated_loss = []
        self.true_loss = []

        # generate list of starting mean sample values
        self.starting_mu_list = self._assign_normal_values(starting_mu_args,
                                                           samples,
                                                           starting_mu_dist,
                                                           absolute=absolute)

        # generate list of ending mean sample values
        self.ending_mu_list = self._assign_normal_values(ending_mu_args,
                                                         samples,
                                                         ending_mu_dist,
                                                         absolute=absolute)

        # generate list of standard deviations for each loss timeseries
        self.std_starting_list = self._assign_normal_values(
            starting_std_args, samples, starting_std_dist, absolute=True)

        self.std_ending_list = self._assign_normal_values(
            ending_std_args, samples, ending_std_dist, absolute=True)

        self.loss_slopes = self._gen_loss_slope(self.starting_mu_list,
                                                self.ending_mu_list,
                                                max_time_steps,
                                                slope_type=loss_curve_shape)
        
        self.std_slopes = self._gen_loss_slope(self.std_starting_list, 
                                                self.std_ending_list,
                                                max_time_steps,
                                                slope_type=std_curve_shape)

    def _assign_normal_values(
            self, config: dict, samples: int, dist: Callable, absolute=True) -> np.array:
        # generate list of values based on given config
        if self._check_normal_config(config) < 0:
            return -1
        mu = config["mu"]
        std = config["std"]
        generated_values = dist(mu, std, samples)

        # take absolute value if specified
        if absolute:
            return np.absolute(generated_values)

        return generated_values

    def _gen_loss_slope(self, y1, y2, x_range, slope_type="expo"):
        if slope_type == "expo":
            lambdas = (np.log(y1) - np.log(y2)) / x_range

        elif slope_type == "linear":
            lambdas = (y2 - y1) / x_range
        return lambdas

    def _check_normal_config(self, config: dict) -> int:
        # generate list of starting mean sample values
        if "mu" not in config:
            raise KeyError("mu not provided in config")
            return -1

        if "std" not in config:
            raise KeyError("std not provided in config")
            return -1

        return 0

    def _get_mean_per_time_step(self, starting, slopes, time_step, slope_type="expo"):
        if slope_type == "expo":
            mu_vals = starting * \
                np.exp(-1 * slopes * time_step)
        elif slope_type == "linear":
            mu_vals = starting + slopes * time_step
        return mu_vals

    def generate_landscape(self):

        simulated_loss = []
        true_loss = []

        for time_step in range(self.max_time_steps):
            cur_loss_mu_values = self._get_mean_per_time_step(
                    self.starting_mu_list, 
                    self.loss_slopes,
                    time_step=time_step,
                    slope_type=self.loss_curve_shape)

            cur_std_values = self._get_mean_per_time_step(
                    self.std_starting_list, 
                    self.std_slopes,
                    time_step=time_step,
                    slope_type=self.loss_curve_shape)
                    
            true_loss.append(cur_loss_mu_values)
            simulated_loss.append(
                np.absolute(
                    self.noise_dist(
                        cur_loss_mu_values,
                        cur_std_values)))

        self.simulated_loss = np.array(simulated_loss)
        self.true_loss = np.array(true_loss)
        return self.simulated_loss


if __name__ == "__main__":
    max_time_steps = 100
    print("Debugging Landscaper")
    landscaper = NormalLossDecayLandscape(
        max_time_steps=max_time_steps, samples=10)
    sim_loss = landscaper.generate_landscape()
    time_range = np.arange(0, max_time_steps)
    print(sim_loss[:,0])
    plt.plot(time_range, sim_loss, alpha=0.1, color="blue")
    plt.plot(time_range, landscaper.true_loss, alpha=0.2, color="red")
    plt.show()

    sns.heatmap(sim_loss)
    plt.show()
