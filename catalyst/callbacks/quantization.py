from typing import Dict, Optional, TYPE_CHECKING, Union
from pathlib import Path

import torch

from catalyst.core import Callback, CallbackNode, CallbackOrder
from catalyst.utils import BestModel, quantize_model

if TYPE_CHECKING:
    from catalyst.core import IRunner


class QuantizationCallback(Callback):
    def __init__(
        self,
        metric: str,
        minimize_metric: bool = True,
        qconfig_spec: Dict = None,
        dtype: Union[str, Optional[torch.dtype]] = "qint8",
        logdir: Union[str, Path] = None,
        filename: str = "quantized.pth",
    ):
        """
        @TODO
        Args:
            metric:
            minimize_metric:
            qconfig_spec:
            dtype:
            logdir:
            filename:
        """
        super().__init__(order=CallbackOrder.External, node=CallbackNode.master)
        self.best_model = BestModel(metric=metric, minimize_metric=minimize_metric)
        self.qconfig_spec = qconfig_spec
        self.dtype = dtype
        if logdir is not None:
            self.filename = Path(logdir) / filename
        else:
            self.filename = filename

    def on_epoch_end(self, runner: "IRunner") -> None:
        self.best_model.add_result(epoch_metrics=runner.epoch_metrics, model=runner.model)

    def on_stage_end(self, runner: "IRunner") -> None:
        model = runner.model.cpu()
        model.load_state_dict(self.best_model.get_best_model_sd())
        q_model = quantize_model(model.cpu(), qconfig_spec=self.qconfig_spec, dtype=self.dtype)
        torch.save(q_model.state_dict(), self.filename)


# # @TODO: make the same API for tracing/onnx/pruning/quantization
# from typing import Dict, Optional, Set, TYPE_CHECKING, Union
# from pathlib import Path
#
# import torch
# from torch import quantization
#
# from catalyst.core.callback import Callback, CallbackOrder
# from catalyst.utils.quantization import save_quantized_model
#
# if TYPE_CHECKING:
#     from catalyst.core.runner import IRunner
#
#
# class DynamicQuantizationCallback(Callback):
#     """Dynamic Quantization Callback
#
#     This callback applying dynamic quantization to the model.
#     """
#
#     def __init__(
#         self,
#         metric: str = "loss",
#         minimize: bool = True,
#         min_delta: float = 1e-6,
#         mode: str = "best",
#         do_once: bool = True,
#         qconfig_spec: Optional[Union[Set, Dict]] = None,
#         dtype: Optional[torch.dtype] = torch.qint8,
#         out_dir: Union[str, Path] = None,
#         out_model: Union[str, Path] = None,
#         backend: str = None,
#     ):
#         """Init method for callback
#
#         Args:
#             metric: Metric key we should trace model based on
#             minimize: Whether do we minimize metric or not
#             min_delta: Minimum value of change for metric to be
#                 considered as improved
#             mode: One of `best` or `last`
#             do_once: Whether do we trace once per stage or every epoch
#             qconfig_spec: torch.quantization.quantize_dynamic
#                 parameter, you can define layers to be quantize
#             dtype: type of the model parameters, default int8
#             out_dir (Union[str, Path]): Directory to save model to
#             out_model (Union[str, Path]): Path to save model to
#                 (overrides `out_dir` argument)
#             backend: defines backend for quantization
#         """
#         super().__init__(order=CallbackOrder.external)
#
#         if mode not in ["best", "last"]:
#             raise ValueError(
#                 f"Unknown `mode` '{mode}'. " f"Must be 'best' or 'last'"
#             )
#
#         self.metric = metric
#         self.mode = mode
#         self.do_once = do_once
#         self.best_score = None
#         self.is_better = None
#         self.first_time = True
#         if minimize:
#             self.is_better = lambda score, best: score <= (best - min_delta)
#         else:
#             self.is_better = lambda score, best: score >= (best + min_delta)
#
#         self.opt_level = None
#
#         if out_model is not None:
#             out_model = Path(out_model)
#         self.out_model = out_model
#
#         if out_dir is not None:
#             out_dir = Path(out_dir)
#         self.out_dir = out_dir
#         self.qconfig_spec = qconfig_spec
#         self.dtype = dtype
#
#         if backend is not None:
#             torch.backends.quantized.engine = backend
#
#     def on_epoch_end(self, runner: "IRunner"):
#         """
#         Performing model quantization on epoch end if condition metric is
#         improved
#
#         Args:
#             runner: current runner
#         """
#         if not self.do_once:
#             if self.mode == "best":
#                 score = runner.valid_metrics[self.metric]
#
#                 if self.best_score is None:
#                     self.best_score = score
#
#                 if self.is_better(score, self.best_score) or self.first_time:
#                     self.best_score = score
#                     quantized_model = quantization.quantize_dynamic(
#                         runner.model.cpu(),
#                         qconfig_spec=self.qconfig_spec,
#                         dtype=self.dtype,
#                     )
#                     save_quantized_model(
#                         model=quantized_model,
#                         logdir=runner.logdir,
#                         checkpoint_name=self.mode,
#                         out_model=self.out_model,
#                         out_dir=self.out_dir,
#                     )
#                     self.first_time = False
#             else:
#                 quantized_model = quantization.quantize_dynamic(
#                     runner.model.cpu(),
#                     qconfig_spec=self.qconfig_spec,
#                     dtype=self.dtype,
#                 )
#                 save_quantized_model(
#                     model=quantized_model,
#                     logdir=runner.logdir,
#                     checkpoint_name=self.mode,
#                     out_model=self.out_model,
#                     out_dir=self.out_dir,
#                 )
#
#     def on_stage_end(self, runner: "IRunner") -> None:
#         """
#         On stage end action.
#
#         Args:
#             runner: runner of your experiment
#         """
#         if self.do_once:
#             quantized_model = quantization.quantize_dynamic(
#                 runner.model.cpu(),
#                 qconfig_spec=self.qconfig_spec,
#                 dtype=self.dtype,
#             )
#             save_quantized_model(
#                 model=quantized_model,
#                 logdir=runner.logdir,
#                 checkpoint_name=self.mode,
#                 out_model=self.out_model,
#                 out_dir=self.out_dir,
#             )
#
#
# __all__ = ["DynamicQuantizationCallback"]
