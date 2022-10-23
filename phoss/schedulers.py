import logging
from ray.tune.schedulers import (
    ASHAScheduler, HyperBandScheduler, PopulationBasedTraining,
    MedianStoppingRule, FIFOScheduler
)
from ray import tune


logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self,
                 scheduler_name: str,
                 scheduler_config: dict):

        self.scheduler_name = scheduler_name
        self.scheduler_config = scheduler_config

    def get_instance(self):
        if self.scheduler_name == 'ASHA':
            max_num_epochs = self.scheduler_config.get('max_t', 100)
            brackets = self.scheduler_config.get('brackets', 3)
            grace_period = self.scheduler_config.get('grace_period', 1)
            reduction_factor = self.scheduler_config.get('reduction_factor', 4)

            return ASHAScheduler(
                max_t=max_num_epochs,
                brackets=brackets,
                grace_period=grace_period,
                reduction_factor=reduction_factor
            )

        elif self.scheduler_name == 'Hyperband':
            print('Using Hyperband')
            max_num_epochs = self.scheduler_config.get('max_t', 100)
            reduction_factor = self.scheduler_config.get('reduction_factor', 4)

            return HyperBandScheduler(
                max_t=max_num_epochs,
                time_attr='training_iteration',
                reduction_factor=reduction_factor
            )

        elif self.scheduler_name == 'Median':
            print('Using Median Rule')
            max_num_epochs = self.scheduler_config.get('max_t', 100)
            return MedianStoppingRule(
                time_attr= 'training_iterations',
                grace_period=0,
                min_samples_required=5,
            )

        elif self.scheduler_name == 'PBT':
            print('Using Population-based Training')
            max_num_epochs = self.scheduler_config.get('max_t', 100)
            num_samples = self.scheduler_config.get('num_samples', 100)
            return PopulationBasedTraining(
                time_attr='training_iteration',
                perturbation_interval = max_num_epochs//20,
                quantile_fraction=0.2,
                resample_probability=0.5,
                hyperparam_mutations={
                    'index' : tune.randint(0, num_samples)
                },
            )

        print('Using random configurations with FIFO scheduler')
        return FIFOScheduler()
