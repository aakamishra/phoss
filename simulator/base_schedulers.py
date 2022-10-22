"""Holds all base classes for schedulers"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Union, List
import numpy as np
import pickle

# TODO(aakamishra) Figure out why trial_runner class is not directly acessible
#from ray.tune.execution import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.tune.experiment import Trial

logger = logging.getLogger(__name__)


class BaseBracket(ABC):
    """Bookkeeping system to track the cutoffs.

    Rungs are created in reversed order so that we can more easily find
    the correct rung corresponding to the current iteration of the result.

    Example:
        >>> trial1, trial2, trial3 = ... # doctest: +SKIP
        >>> b = _Bracket(1, 10, 2, 0) # doctest: +SKIP
        >>> # CONTINUE
        >>> b.on_result(trial1, 1, 2) # doctest: +SKIP
        >>> # CONTINUE
        >>> b.on_result(trial2, 1, 4) # doctest: +SKIP
        >>> # rungs are reversed
        >>> b.cutoff(b._rungs[-1][1]) == 3.0 # doctest: +SKIP
         # STOP
        >>> b.on_result(trial3, 1, 1) # doctest: +SKIP
        >>> b.cutoff(b._rungs[3][1]) == 2.0 # doctest: +SKIP
    """

    def __init__(
        self,
        min_trials: int,
        max_trials: int,
        reduction_factors: List[int],
        starting_rate: int,
        stop_last_trials: bool = True
    ):
        assert len(reduction_factors) > 0
        self.reduction_factor1 = reduction_factors[0]
        self.reduction_factor2 = reduction_factors[1] if len(reduction_factors) == 2 else 0
        self.starting_rate = starting_rate
        self.MAX_RUNGS = int(np.log(max_trials / min_trials) / np.log(self.reduction_factor1) - starting_rate + 1)
        self.rungs = [
            (min_trials * self.reduction_factor1 ** (k + starting_rate), {}) for k in reversed(range(self.MAX_RUNGS))
        ]
        self.stop_last_trials = stop_last_trials

    @abstractmethod
    def cutoff(
        self, recorded, index=None
    ) -> Optional[Union[int, float, complex, np.ndarray]]:
        pass

    def rung_visualizer(self):
        print('Bracket', self.starting_rate)
        for rung in self.rungs:
            val_list = [
                '( ' + str(key[:-4]) + ' || ' + '%.2f )' % elem
                for key, elem in rung[1].items()
            ]
            print('Rsrc: ', rung[0], 'cont: ', val_list)

    def on_result(self, trial: Trial, cur_iter: int, cur_rew: Optional[float]) -> str:
        action = TrialScheduler.CONTINUE
        for i, milestone, recorded in enumerate(self.rungs):
            if (
                cur_iter >= milestone
                and trial.trial_id in recorded
                and not self.stop_last_trials
            ):
                # If our result has been recorded for this trial already, the
                # decision to continue training has already been made. Thus we can
                # skip new cutoff calculation and just continue training.
                # We can also break as milestones are descending.
                break
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                cutoff = self.cutoff(recorded, index=i)
                if cutoff is not None and cur_rew < cutoff:
                    action = TrialScheduler.STOP
                if cur_rew is None:
                    logger.warning(
                        "Reward attribute is None! Consider"
                        " reporting using a different field."
                    )
                else:
                    recorded[trial.trial_id] = cur_rew
                break
        print("Bracket s: ", self.starting_rate, "details: ", self.debug_str())
        return action

    def debug_str(self) -> str:
        # TODO: fix up the output for this
        iters = " | ".join(
            [
                "Iter {:.3f}: {}".format(milestone, self.cutoff(recorded))
                for milestone, recorded in self.rungs
            ]
        )
        return "Bracket: " + iters


class BaseAsyncHyperBandScheduler(FIFOScheduler):
    """Implements the Async Successive Halving.

    This should provide similar theoretical performance as HyperBand but
    avoid straggler issues that HyperBand faces. One implementation detail
    is when using multiple brackets, trial allocation to bracket is done
    randomly with over a softmax probability.

    See https://arxiv.org/abs/1810.05934

    Args:
        time_attr: A training result attr to use for comparing time.
            Note that you can pass in something non-temporal such as
            `training_iteration` as a measure of progress, the only requirement
            is that the attribute should increase monotonically.
        metric: The training result objective value attribute. Stopping
            procedures will use this attribute. If None but a mode was passed,
            the `ray.tune.result.DEFAULT_METRIC` will be used per default.
        mode: One of {min, max}. Determines whether objective is
            minimizing or maximizing the metric attribute.
        max_t: max time units per trial. Trials will be stopped after
            max_t time units (determined by time_attr) have passed.
        grace_period: Only stop trials at least this old in time.
            The units are the same as the attribute named by `time_attr`.
        reduction_factor: Used to set halving rate and amount. This
            is simply a unit-less scalar.
        brackets: Number of brackets. Each bracket has a different
            halving rate, specified by the reduction factor.
        stop_last_trials: Whether to terminate the trials after
            reaching max_t. Defaults to True.
    """

    def __init__(
        self,
        time_attr: str = "training_iteration",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        max_t: int = 100,
        grace_period: int = 1,
        reduction_factors: List[float] = [4],
        brackets: int = 1,
        stop_last_trials: bool = True,
        bracket_class: BaseBracket = BaseBracket,
    ):
        assert max_t > 0, "Max (time_attr) not valid!"
        assert max_t >= grace_period, "grace_period must be <= max_t!"
        assert grace_period > 0, "grace_period must be positive!"
        # assert reduction_factor > 1, "Reduction Factor not valid!"
        assert brackets > 0, "brackets must be positive!"
        if mode:
            assert mode in ["min", "max"], "`mode` must be 'min' or 'max'!"

        FIFOScheduler.__init__(self)
        self._reduction_factors = reduction_factors
        self._max_t = max_t

        self._trial_info = {}  # Stores Trial -> Bracket

        # Tracks state for new trial add
        self._brackets = [
            bracket_class(
                grace_period,
                max_t,
                reduction_factors,
                s,
                stop_last_trials=stop_last_trials,
            )
            for s in range(brackets)
        ]
        self._counter = 0  # for
        self._num_stopped = 0
        self._metric = metric
        self._mode = mode
        self._metric_op = None
        if self._mode == "max":
            self._metric_op = 1.0
        elif self._mode == "min":
            self._metric_op = -1.0
        self._time_attr = time_attr
        self._stop_last_trials = stop_last_trials

    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], **spec
    ) -> bool:
        if self._metric and metric:
            return False
        if self._mode and mode:
            return False

        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        if self._mode == "max":
            self._metric_op = 1.0
        elif self._mode == "min":
            self._metric_op = -1.0

        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC

        return True

    def on_trial_add(self, trial_runner, trial: Trial):
        if not self._metric or not self._metric_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.TuneConfig()`".format(
                    self.__class__.__name__, self._metric, self._mode
                )
            )

        sizes = np.array([len(b.rungs) for b in self._brackets])
        probs = np.e ** (sizes - sizes.max())
        normalized = probs / probs.sum()
        idx = np.random.choice(len(self._brackets), p=normalized)
        self._trial_info[trial.trial_id] = self._brackets[idx]

    def on_trial_result(
        self, trial_runner, trial: Trial, result: Dict
    ) -> str:
        action = TrialScheduler.CONTINUE
        if self._time_attr not in result or self._metric not in result:
            return action
        if result[self._time_attr] >= self._max_t and self._stop_last_trials:
            action = TrialScheduler.STOP
        else:
            bracket = self._trial_info[trial.trial_id]
            action = bracket.on_result(
                trial, result[self._time_attr], self._metric_op * result[self._metric]
            )
        if action == TrialScheduler.STOP:
            self._num_stopped += 1
        return action

    def on_trial_complete(
        self, trial_runner, trial: Trial, result: Dict
    ):
        if self._time_attr not in result or self._metric not in result:
            return
        bracket = self._trial_info[trial.trial_id]
        bracket.on_result(
            trial, result[self._time_attr], self._metric_op * result[self._metric]
        )
        del self._trial_info[trial.trial_id]

    def on_trial_remove(self, trial_runner, trial: Trial):
        del self._trial_info[trial.trial_id]

    def debug_string(self) -> str:
        out = "Using AsyncHyperBand: num_stopped={}".format(self._num_stopped)
        out += "\n" + "\n".join([b.debug_str() for b in self._brackets])
        return out

    def save(self, checkpoint_path: str):
        save_object = self.__dict__
        with open(checkpoint_path, "wb") as outputFile:
            pickle.dump(save_object, outputFile)

    def restore(self, checkpoint_path: str):
        with open(checkpoint_path, "rb") as inputFile:
            save_object = pickle.load(inputFile)
        self.__dict__.update(save_object)
