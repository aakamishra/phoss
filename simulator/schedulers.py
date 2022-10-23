from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler, PopulationBasedTraining, TrialScheduler, MedianStoppingRule, FIFOScheduler
from ray import tune
import logging
from typing import Dict, Optional, Union, List

import numpy as np


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

        elif self.scheduler_name == "Hyperband":
            print("using Hyperband")
            max_num_epochs = self.scheduler_config.get("max_t", 100)
            reduction_factor = self.scheduler_config.get("reduction_factor", 4)

            return HyperBandScheduler(max_t=max_num_epochs,
                                    time_attr="training_iteration",
                                    reduction_factor=reduction_factor)

        elif self.scheduler_name == "Median":
            print("using Median Rule")
            max_num_epochs = self.scheduler_config.get("max_t", 100)
            return MedianStoppingRule(time_attr= 'training_iteration',
                                    grace_period=2, 
                                    min_samples_required=3,
                                    min_time_slice = 0,
                                    )
        
        elif self.scheduler_name == "PBT":
            print("using Population-based Training")
            max_num_epochs = self.scheduler_config.get("max_t", 100)
            num_samples = self.scheduler_config.get("num_samples", 100)
            return PopulationBasedTraining(
                        time_attr="training_iteration",
                        perturbation_interval = max_num_epochs//5, 
                        quantile_fraction=0.2,
                        resample_probability=0.2,
                        hyperparam_mutations={
                            "index" : tune.randint(0, num_samples)
                        },
                    )
        else:
            print("running Random Configurations")
            return FIFOScheduler()

