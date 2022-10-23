import numpy as np
from ray import tune


class SimulatedTrainable(tune.Trainable):
    """Class for simulating a PyTorch Lightning Module trainable"""

    def setup(self, config):
        """
        Trainable Class Setup function for intitalizing module with values.

        Args
        ----
        config: dict, contains key with the index of the assigned loss curve and
        the values array

        Returns
        ------
        None
        """
        self.index = config.get('index', 0)

        self.file = config.get('gen_sim_path', '')
        self.values = np.genfromtxt(self.file, delimiter=',')
        self.series = self.values[:,self.index]
        self.internal_training_iteration = 0
        self.verbose = config.get('verbose', False)

    def reset_config(self, new_config: dict) -> bool:
        """
        Function for resetting an actor in order for reuse_actors=True
        for the Tune config. This is a more efficient method of setting up
        new workers.

        Args
        ----
        new_config: dict, should contain the index of the next assigned curve

        Returns
        -------
        bool: True if the function resets the actor correctly and assigns
        a new series to be trained on.
        """
        if 'index' in new_config:
            self.index = new_config.get('index')
            self.series = self.values[:,self.index]
            self.internal_training_iteration = 0
            return True
        else:
            return False

    def step(self):
        """
        Runs one step of the training module by
        pulling a value from the generated loss curve
        depending on the current iteration.

        Args
        ----
        None

        Returns
        -------
        None
        """
        loss = self.series[self.training_iteration]
        self.internal_training_iteration += 1
        return {'index': self.index, 'loss': loss}

    def save_checkpoint(self, checkpoint_dir: str) -> None:
        """
        No Model to checkpoint
        """
        if self.verbose:
            print('Virtual Save:', checkpoint_dir)

    def load_checkpoint(self, checkpoint: str) -> None:
        """
        No Models to load
        """
        if self.verbose:
            print('Virtual Load:', checkpoint)
