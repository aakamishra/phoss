import json
import numpy as np


DEFAULT_STARTING_STD_ARGS = {'mu': 0.02, 'std': 0.005}
DEFAULT_ENDING_STD_ARGS = {'mu': 0.001, 'std': 0.005}


class ParametricConfig:

    def __init__(self, beta1, b1std, beta2, b2std, beta3, b3std, alpha,
                 alphastd):
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.alpha = alpha
        self.b1std = b1std
        self.b2std = b2std
        self.b3std = b3std
        self.alphastd = alphastd

    @staticmethod
    def read_from_dict(values: dict):
        beta1 = values.get('beta1', 0)
        beta2 = values.get('beta2', 0)
        beta3 = values.get('beta3', 0)

        b1std = values.get('b1std', 0)
        b2std = values.get('b2std', 0)
        b3std = values.get('b3std', 0)

        alpha = values.get('alpha', 0)
        alphastd = values.get('alphastd', 0)

        return ParametricConfig(beta1, b1std, beta2, b2std, beta3, b3std, alpha,
                                alphastd)

class ParametricLossCurve:

    def __init__(self, json_name=None, num_samples=10, beta0=0, b0std=0.01,
                 seed=109, max_num_epochs=100):
        self.beta0 = beta0
        self.b0std = b0std
        self.max_num_epochs = max_num_epochs
        self.config_list = []
        self.num_samples = num_samples
        self.std_start = DEFAULT_STARTING_STD_ARGS
        self.std_end = DEFAULT_ENDING_STD_ARGS
        self.json_name = json_name
        self.seed = seed
        np.random.seed(self.seed)

        with open(json_name, encoding='utf-8') as f:
            configs = json.load(f)
            self.beta0 = configs['beta0']
            self.b0std = configs['b0std']
            self.std_start = configs['std_start']
            self.std_end = configs['std_end']
            dict_config_list = configs['config_list']
            for cf in dict_config_list:
                self.config_list.append(ParametricConfig.read_from_dict(cf))

        self.random_sample_config_list = []
        for config in self.config_list:
            rn_cf = {}
            rn_cf['beta1'] = np.random.normal(
                loc=config.beta1,
                scale=config.b1std,
                size=num_samples
            )
            rn_cf['beta2'] = np.random.normal(
                loc=config.beta2,
                scale=config.b2std,
                size=num_samples
            )
            rn_cf['beta3'] = np.random.normal(
                loc=config.beta3,
                scale=config.b3std,
                size=num_samples
            )
            alpha_array = np.random.normal(
                loc=config.alpha,
                scale=config.alphastd,
                size=num_samples
            )
            np.clip(alpha_array, config.alpha - config.alphastd,
                    config.alpha + config.alphastd, out=alpha_array)
            rn_cf['alpha'] = alpha_array
            self.random_sample_config_list.append(rn_cf)
        self.beta0_list = np.random.normal(loc=self.beta0, scale=self.b0std,
                                           size=self.num_samples)


    def generate_curve_means(self, time):
        time = np.array([time])
        time_series = np.repeat(time, self.num_samples, axis=0)

        total = self.beta0_list
        vals = []
        for config in self.random_sample_config_list:
            beta1 = config['beta1']
            beta2 = config['beta2']
            beta3 = config['beta3']
            alpha = config['alpha']
            vals.append(
                beta1 / ( np.float_power(time_series + beta2, alpha) + beta3))

        return total + np.sum(np.array(vals), axis=0)

    def generate(self):
        vals = []
        for i in range(self.max_num_epochs):
            vals.append(self.generate_curve_means(i))
        return np.array(vals)


class NormalLossDecayLandscape:
    def __init__(
        self,
        json_config,
        max_time_steps: int = 100,
        samples: int = 50,
        seed: int = 109,
        absolute: bool = True,
        std_curve_shape: str = 'linear',
    ):

        # set random seed for experiment replicability
        self.seed = seed
        np.random.seed(self.seed)

        # store instance values
        self.absolute = absolute
        self.samples = samples
        self.max_time_steps = max_time_steps
        self.std_curve_shape = std_curve_shape
        self.json_config = json_config
        self.simulated_loss = []
        self.true_loss = []


        self.parametric_curve = ParametricLossCurve(
            json_name=self.json_config,
            num_samples=self.samples,
            seed=self.seed,
            max_num_epochs=self.max_time_steps
        )
        # generate list of standard deviations for each loss timeseries
        self.std_starting_list = self._assign_normal_values(
            self.parametric_curve.std_start, samples, absolute=True)

        self.std_ending_list = self._assign_normal_values(
            self.parametric_curve.std_end, samples, absolute=True)


        self.std_slopes = self._gen_loss_slope(
            self.std_starting_list,
            self.std_ending_list,
            max_time_steps,
            slope_type=std_curve_shape
        )

    def _assign_normal_values(
        self, config: dict, samples: int, absolute=True
    ) -> np.array:
        # generate list of values based on given config
        if self._check_normal_config(config) < 0:
            return -1
        mu = config['mu']
        std = config['std']
        generated_values = np.random.normal(mu, std, samples)

        # take absolute value if specified
        if absolute:
            return np.absolute(generated_values)

        return generated_values

    def _gen_loss_slope(self, y1, y2, x_range, slope_type='expo'):
        if slope_type == 'expo':
            lambdas = (np.log(y1) - np.log(y2)) / x_range

        elif slope_type == 'linear':
            lambdas = (y2 - y1) / x_range
        return lambdas

    def _check_normal_config(self, config: dict) -> int:
        # generate list of starting mean sample values
        if 'mu' not in config:
            raise KeyError('mu not provided in config')
        if 'std' not in config:
            raise KeyError('std not provided in config')
        return 0

    def _get_mean_per_time_step(
        self,
        starting,
        slopes,
        time_step,
        slope_type='expo'
    ):
        if slope_type == 'expo':
            mu_vals = starting * np.exp(-1 * slopes * time_step)
        elif slope_type == 'linear':
            mu_vals = starting + slopes * time_step
        return mu_vals

    def generate_landscape(self):
        lower_bound = 0
        upper_bound = 1
        simulated_loss = []
        true_loss = []

        loss_mu_values = self.parametric_curve.generate()
        for time_step in range(self.max_time_steps):
            cur_loss_mu_values = loss_mu_values[time_step]

            cur_std_values = self._get_mean_per_time_step(
                    self.std_starting_list,
                    self.std_slopes,
                    time_step=time_step,
                    slope_type=self.std_curve_shape)

            true_loss.append(cur_loss_mu_values)

            simulated_loss.append(
                    np.random.normal(
                        cur_loss_mu_values,
                        cur_std_values))

        self.simulated_loss = np.array(simulated_loss)
        np.clip(self.simulated_loss, lower_bound, upper_bound,
                out=self.simulated_loss)
        self.true_loss = np.array(true_loss)
        np.clip(self.true_loss, lower_bound, upper_bound, out=self.true_loss)
        return self.simulated_loss
