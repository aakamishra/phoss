from ray.tune.schedulers import ASHAScheduler, HyperBandScheduler, PopulationBasedTraining, TrialScheduler

import logging
from typing import Dict, Optional, Union, List

import numpy as np
from ray.tune.experiment import Trial

from base_schedulers import BaseBracket, BaseAsyncHyperBandScheduler

logger = logging.getLogger(__name__)


class PredASHABracket(BaseBracket):

    def cutoff(
        self, recorded, index=None
    ) -> Optional[Union[int, float, complex, np.ndarray]]:
        if not recorded:
            return None
        if not index or index == (self.MAX_RUNGS - 1):
            return np.nanpercentile(list(recorded.values()),
                                    (1 - 1 / self.reduction_factor1) * 100)
        else:
            y1 = np.array(self.rungs[index - 1][1].values())
            print('y1: ', y1)
            x_range = self.rungs[index][0] - self.rungs[index - 1][0]
            print('x-range: ', x_range)
            y2 = recorded.values()
            lambdas = (np.log(y1) - np.log(y2)) / x_range
            print(lambdas)
            future_x = x_range * self.reduction_factor1
            vals = y1 * np.exp(-1 * lambdas * future_x)
            return np.nanpercentile(vals,
                                    (1 - 1 / self.reduction_factor1) * 100)

    def on_result(self, trial: Trial, cur_iter: int,
                  cur_rew: Optional[float]) -> str:
        action = TrialScheduler.CONTINUE
        for i, milestone, recorded in enumerate(self.rungs):
            if (cur_iter >= milestone and trial.trial_id in recorded and
                not self.stop_last_trials):
                # If our result has been recorded for this trial already, the
                # decision to continue training has already been made. Thus we
                # can skip new cutoff calculation and just continue training.
                # We can also break as milestones are descending.
                break
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                cutoff = self.cutoff(recorded, index=i)
                pred_cur_rew = cur_rew
                if i != self.MAX_RUNGS - 1:
                    y1 = np.array(self.rungs[i - 1][1][trial.trial_id])
                    x_range = self.rungs[i][0] - self.rungs[i - 1][0]
                    y2 = recorded
                    slope = (np.log(y1) - np.log(y2)) / x_range
                    future_x = x_range * self.reduction_factor1
                    pred_cur_rew = y1 * np.exp(-1 * slope * future_x)
                    print(
                        'print prediction based: ',
                        pred_cur_rew,
                        cur_rew,
                        cutoff)
                if cutoff is not None and pred_cur_rew < cutoff:
                    action = TrialScheduler.STOP
                if cur_rew is None:
                    logger.warning(
                        'Reward attribute is None! Consider'
                        ' reporting using a different field.'
                    )
                else:
                    recorded[trial.trial_id] = cur_rew
                break
        print('Bracket s: ', self.starting_rate, 'details: ', self.debug_str())
        print(
            'Bracket s: {}, details: {}'.format(self.starting_rate,
                                                self.debug_str())
        )
        return action


class PredAsyncHyperBandScheduler(BaseAsyncHyperBandScheduler):

    def __init__(
        self,
        time_attr: str = 'training_iteration',
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factor: float = 4,
        brackets: int = 1,
        stop_last_trials: bool = True,
    ):
        super(PredAsyncHyperBandScheduler, self).__init__(
            time_attr=time_attr,
            metric=metric,
            mode=mode,
            max_t=max_t,
            grace_period=grace_period,
            reduction_factors=[reduction_factor],
            brackets=brackets,
            stop_last_trials=stop_last_trials,
            bracket_class=PredASHABracket
        )


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

        elif self.scheduler_name == 'PredASHA':
            print('using PredASHA')
            max_num_epochs = self.scheduler_config.get('max_t', 100)
            brackets = self.scheduler_config.get('brackets', 3)
            grace_period = self.scheduler_config.get('grace_period', 1)
            reduction_factor = self.scheduler_config.get('reduction_factor', 4)

            return PredASHAScheduler(
                max_t=max_num_epochs,
                brackets=brackets,
                grace_period=grace_period,
                reduction_factor=reduction_factor
            )

        elif self.scheduler_name == 'Hyperband':
            print('using Hyperband')
            max_num_epochs = self.scheduler_config.get('max_t', 100)
            reduction_factor = self.scheduler_config.get('reduction_factor', 4)

            return HyperBandScheduler(max_t=max_num_epochs,
                                      time_attr='training_iteration',
                                      reduction_factor=reduction_factor)


PredASHAScheduler = PredAsyncHyperBandScheduler

if __name__ == '__main__':
    sched = PredAsyncHyperBandScheduler(
        grace_period=1, max_t=10, reduction_factor=2)
    print(sched.debug_string())
    bracket = sched._brackets[0]
    print(bracket.cutoff({str(i): i for i in range(20)}))
