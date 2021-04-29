from typing import Dict
import numpy as np
from catalyst.core.logger import ILogger
from catalyst.settings import SETTINGS

if SETTINGS.comet_required:
    try:
        from comet_ml import Experiment
        comet_installed = True
    except:
        comet_installed = False 


class CometLogger(ILogger):
    def __init__(self):
        global comet_installed
        self._logging = False
        if comet_installed:
            try:
                self._experiment = Experiment()
                self._logging = True
            except Exception as e:
                print(e)
                print("Comet not configured properly")
        else:
            print("Comet is not installed")

    def log_metrics(
        self,
        metrics: Dict[str, float],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        self._experiment.log_metrics(metrics, step=global_batch_step)

    def log_image(
        self,
        tag: str,
        image: np.ndarray,
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs image to the logger."""
        self._experiment.log_image(image, "images")

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        """Logs hyperparameters to the logger."""
        self._experiment.log_parameters(hparams)

    def close_log(self) -> None:
        """Closes the logger."""
        self._experiment.end()

__all__ = ["CometLogger"]
