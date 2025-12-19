# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import hashlib
import yaml
import logging
import tempfile
from importlib.metadata import distributions
from collections.abc import Callable, Hashable
from functools import partial
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import numpy as np
import numpy.typing as ntp
import torch
import lightning as L
from lightning.pytorch.utilities.seed import _collect_rng_states  # type: ignore[attr-defined]

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.pytorch.schedule import (
    Schedule,
    Never,
)
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger

logger: logging.Logger = logging.getLogger(__name__)

PATH_PREFIX = "introspection"
DTYPES = {
    "float32": "fp32",
    "float64": "fp64",
    "float16": "fp16",
    "bfloat16": "bf16",
    "int64": "i64",
    "int32": "i32",
    "int16": "i16",
    "int8": "i8",
    "uint8": "u8",
    "bool": "b",
    "complex64": "c64",
    "complex128": "c128",
    "qint8": "qi8",
    "quint8": "qui8",
    "qint32": "qi32",
    "float8_e4m3fn": "fp8_e4m3fn",
    "float8_e4m3fnuz": "fp8_e4m3fnuz",
    "float8_e5m2": "fp8_e5m2",
    "float8_e5m2fnuz": "fp8_e5m2fnuz",
    "float8_e8m0fnu": "fp8_e8m0fnu",
}


def _get_dtype(dtype: torch.dtype) -> str:
    for k, v in DTYPES.items():
        if getattr(torch, k) == dtype:
            return v
    return str(dtype)


def _process_tensor_or_ndarray(
    item: Any, tensor_stats: set[str], path: str, parent_dict: dict[str, Any] | None
) -> bytes | None:
    """Process tensor or ndarray items and calculate checksums.

    Args:
        item: The tensor or ndarray to process
        tensor_stats (set[str]): Tensor stats to collect
        path: Current path in the nested structure
        parent_dict: Dictionary to store checksums in

    Returns:
        bytes | None: Digest of the checksum
    """
    if isinstance(item, torch.Tensor):
        cks = tensor_checksum(item)
    else:  # np.ndarray
        cks = numpy_checksum(item)

    if parent_dict is not None:
        parent_dict[path] = _format(item, cks.hexdigest(), tensor_stats)
    return cks.digest()


def _process_nested(sub_item: Any, tensor_stats: set[str], nested_cks: Any, fn: Callable[[Any], Any]) -> None:
    nested: dict[str, Any] = {}
    nested_key = None
    if isinstance(sub_item, torch.Tensor | np.ndarray):
        digest = _process_tensor_or_ndarray(sub_item, tensor_stats, (nested_key := "temp"), nested)
    elif isinstance(sub_item, list | tuple | set | frozenset) or hasattr(sub_item, "items"):
        digest = _process_collection(sub_item, tensor_stats, (nested_key := "temp"), nested)
    if nested_key:
        processed = None
        if digest is not None:
            nested_cks.update(digest)
            processed = nested[nested_key]
        fn(processed)
    else:
        fn(sub_item)
        digest = str(sub_item).encode("utf-8")
        nested_cks.update(digest)


def _ensure_hashable(item: Any) -> Hashable:
    if isinstance(item, Hashable):
        return item
    if isinstance(item, list | tuple):
        return tuple(_ensure_hashable(i) for i in item)
    if isinstance(item, set | frozenset):
        return frozenset(_ensure_hashable(i) for i in item)
    return str(item)


def _process_collection(
    item: Any,
    tensor_stats: set[str],
    path: str,
    parent_dict: dict[str, Any] | None,
) -> bytes | None:
    """Process collection items (list, tuple, dict) and calculate checksums.
    Maintains the nested structure of collections while replacing only tensor values with hashes.
    Primitive values are kept as is.

    Args:
        item: The collection to process
        tensor_stats (set[str]): Tensor stats to collect
        path: Current path in the nested structure
        parent_dict: Dictionary to store checksums in

    Returns:
        bytes | None: Digest of the overall checksum
    """
    nested_cks = hashlib.md5()
    result: list[Any] | dict[Any, Any] | None = None
    if hasattr(item, "items"):  # map-like
        result = {}
        for sub_key, sub_item in item.items():
            nested: list[str] = []
            _process_nested(sub_key, tensor_stats, nested_cks, nested.append)
            key = _ensure_hashable(nested.pop())
            _process_nested(sub_item, tensor_stats, nested_cks, partial(result.__setitem__, key))
    elif isinstance(item, list | tuple | set | frozenset):
        result = []
        for sub_item in item:
            _process_nested(sub_item, tensor_stats, nested_cks, result.append)
    if result is not None and parent_dict is not None and path:
        parent_dict[path] = result
    return nested_cks.digest()


def process_item(
    item: Any, tensor_stats: set[str], path: str = "", parent_dict: dict[str, Any] | None = None
) -> bytes | None:
    """Recursively process items in the batch and calculate checksums.
    Only tensor values are replaced with hashes, primitive values are kept as is.

    Args:
        item: The item to process
        tensor_stats (set[str]): Tensor stats to collect
        path: Current path in the nested structure
        parent_dict: Dictionary to store checksums in

    Returns:
        bytes | None: Digest of the checksum if available, None otherwise
    """
    if isinstance(item, torch.Tensor | np.ndarray):
        return _process_tensor_or_ndarray(item, tensor_stats, path, parent_dict)
    elif isinstance(item, list | tuple | set | frozenset) or hasattr(item, "items"):
        return _process_collection(item, tensor_stats, path, parent_dict)
    elif not isinstance(item, str | int | float | bool) or item is not None:
        logging.warning(f"Converting {type(item).__name__} to string for checksum")
        item = str(item)
    if parent_dict is not None:
        parent_dict[path] = item
    return str(item).encode("utf-8")


def tensor_checksum(tensor: torch.Tensor) -> Any:
    """Tensor checksum hash, returns the same value for tensors with identical contents

    Args:
        tensor (torch.Tensor): tensor to generate the checksum

    Return:
        hashlib.md5: checksum hash
    """
    tensor = tensor.to(torch.float32) if tensor.dtype == torch.bfloat16 else tensor
    return numpy_checksum(tensor.detach().cpu().numpy())


def numpy_checksum(ndarray: ntp.NDArray[Any]) -> Any:
    """Numpy NDArray checksum, returns the same value for ndarrays with identical contents

    Args:
        tensor (torch.Tensor): tensor to generate the checksum

    Return:
        hashlib.md5: checksum hash
    """
    return hashlib.md5(ndarray.tobytes())


def _params_checksums(
    model: torch.nn.Module,
    params_checksum: bool,
    grads_checksum: bool,
    gradients: dict[str, Any],
    tensor_stats: set[str],
) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    grads_cks = hashlib.md5()
    params_cks = hashlib.md5()
    for name, param in model.named_parameters():
        parameters[name] = {}
        if params_checksum and param.data is not None:
            param_cks = tensor_checksum(param.data)
            parameters[name]["data"] = _format(param.data, param_cks.hexdigest(), tensor_stats)
            params_cks.update(param_cks.digest())
        grad_cks, repr = gradients.get(name, (None, None))
        if grads_cks is not None:
            assert grad_cks
            parameters[name]["grad"] = repr
            grads_cks.update(grad_cks.digest())
    if params_checksum:
        _add_digest(parameters, "__all_data__", params_cks)
    if grads_checksum:
        _add_digest(parameters, "__all_grads__", grads_cks)
    return parameters


def _format(tensor: torch.Tensor, hash: str, tensor_stats: set[str]) -> str:  # noqa: C901
    if tensor.dim() == 0:
        return str(tensor)
    chunks: list[str] = []
    if "shape" in tensor_stats:
        chunks.append("×".join(str(d) for d in tensor.shape))
    if "dtype" in tensor_stats:
        chunks.append(_get_dtype(tensor.dtype))
    if "infs" in tensor_stats:
        num_pos_infs = tensor.isposinf().sum()
        chunks.append(f"{num_pos_infs}∞̟")
        num_neg_infs = tensor.isneginf().sum()
        chunks.append(f"{num_neg_infs}∞̠")
    if "nans" in tensor_stats:
        num_nans = tensor.isnan().sum()
        chunks.append(f"{num_nans}⚠")
    if "zeros" in tensor_stats:
        num_zeros = tensor.numel() - tensor.count_nonzero().item()
        chunks.append(f"{num_zeros}⌀")
    if "med" in tensor_stats:
        med = tensor.median().item()
        chunks.append(f"{med}m̃")
    if "mean" in tensor_stats:
        mean = tensor.float().mean().item()
        chunks.append(f"{mean}μ")
    if "amean" in tensor_stats:
        amean = tensor.abs().float().mean().item()
        chunks.append(f"{amean}μ⁺")
    if "std" in tensor_stats:
        std = tensor.float().std().item()
        chunks.append(f"{std}σ")
    if "var" in tensor_stats:
        var = tensor.float().var(unbiased=False).item()
        chunks.append(f"{var}σ²")
    if "uvar" in tensor_stats:
        uvar = tensor.float().var(unbiased=True).item()
        chunks.append(f"{uvar}s²")
    if "skew" in tensor_stats:
        t = tensor.float()
        std = t.std().item()
        skew = ((t - t.mean()) / std).pow(3).double().mean().item() if std != 0 else "?"
        chunks.append(f"{skew}γ₁")
    if "kurt" in tensor_stats:
        t = tensor.float()
        std = t.std().item()
        kurt = ((t - t.mean()) / std).pow(4).double().mean().item() if std != 0 else "?"
        chunks.append(f"{kurt}γ₂")
    if "mode" in tensor_stats:
        vals, counts = torch.unique(tensor, return_counts=True)
        _, idx = counts.max(0)
        mode = vals[idx]
        chunks.append(f"{mode}Mo")
    if "min" in tensor_stats:
        mi = tensor.min().item()
        chunks.append(f"{mi}↤")
    if "max" in tensor_stats:
        ma = tensor.max().item()
        chunks.append(f"{ma}↦")
    if "hash" in tensor_stats:
        chunks.append(hash)
    res = "|".join(chunks)
    return res


def _buffers_checksums(model: torch.nn.Module, tensor_stats: set[str]) -> dict[str, Any]:
    buffers = {}
    buffers_cks = hashlib.md5()
    for name, buffer in model.named_buffers():
        if buffer is not None:
            buffers_cks = tensor_checksum(buffer)
            buffers[name] = _format(buffer, buffers_cks.hexdigest(), tensor_stats)
            buffers_cks.update(buffers_cks.digest())
    _add_digest(buffers, "__all_buffers__", buffers_cks)
    return buffers


def _rngs_checksums() -> dict[str, Any]:
    rngs: dict[str, Any] = {}
    rng_states = _collect_rng_states(include_cuda=True)  # type: ignore[attr-defined]
    for name, state in rng_states.items():
        if name == "torch":
            cks = tensor_checksum(state).hexdigest()
        elif name == "torch.cuda":
            cks = tensor_checksum(torch.stack(state, dim=0)).hexdigest()
        elif name == "numpy":
            cks = {
                "algo": state[0],
                "state": numpy_checksum(state[1]).hexdigest(),
                "pos": state[2],
                "has_gauss": state[3],
                "cached_gaussian": state[4],
            }
        elif name == "python":
            cks = {
                "version": state[0],
                "state": numpy_checksum(np.array(state[1])).hexdigest(),
                "gaussian": state[2],
            }
        else:
            raise RuntimeError(f"Unsupported RNG state '{name}': {state}")
        rngs[name] = cks
    return rngs


def get_checksums(
    trainer: L.Trainer,
    model: torch.nn.Module,
    gradients: dict[str, Any],
    checksums: set[str],
    tensor_stats: set[str],
) -> dict[str, Any]:
    """Checksums for internal model state and training context.

    Args:
        model (torch.nn.Module): model to generate the parameters checksum
        checksums (set[str]): - checksums to collect

    Returns:
        dict: captured checksums
    """
    cks: dict[str, Any] = {}
    params_checksum = "params" in checksums
    grads_checksum = "grads" in checksums
    if params_checksum or grads_checksum:
        cks["parameters"] = _params_checksums(model, params_checksum, grads_checksum, gradients, tensor_stats)
    if "buffers" in checksums:
        cks["buffers"] = _buffers_checksums(model, tensor_stats)
    if "optimizers" in checksums:
        cks["optimizers"] = _optimizers_checksums(trainer, model, tensor_stats)
    if "rngs" in checksums:
        cks["rngs"] = _rngs_checksums()
    return cks


def _batch_checksums(batch: Any, batch_idx: int, tensor_stats: set[str]) -> dict[str, Any]:
    """Calculate checksums for batch data.
    Args:
        batch (Any): The batch data to generate checksums for
        batch_idx (int): The batch index
    Returns:
        dict[str, Any]: Dictionary containing checksums for batch elements
    """
    checksums: dict[str, Any] = {"__batch_idx__": batch_idx}
    _add_digest(checksums, "__all_batch__", process_item(batch, tensor_stats, "__batch__", checksums))
    return checksums


def _add_digest(cks: dict[str, Any], name: str, digest: Any) -> None:
    if digest is not None:
        cks[name] = digest.hex() if isinstance(digest, bytes) else digest.hexdigest()


def _optimizers_checksums(trainer: L.Trainer, pl_module: torch.nn.Module, tensor_stats: set[str]) -> list[Any]:
    """Extract and compute checksums of optimizer states"""
    checksums = []
    for opt in trainer.optimizers:
        opt_cks = {"__type__": type(opt).__name__}
        _add_digest(opt_cks, "__all_defaults__", process_item(opt.defaults, tensor_stats, "defaults", opt_cks))
        _add_digest(
            opt_cks, "__all_param_groups__", process_item(opt.param_groups, tensor_stats, "param_groups", opt_cks)
        )
        _add_digest(opt_cks, "__all_state__", process_item(opt.state, tensor_stats, "state", opt_cks))
        checksums.append(opt_cks)
    return checksums


class Introspection(L.Callback):
    def __init__(
        self,
        checksums: set[str] | None = None,
        tensor_stats: set[str] | None = None,
        env_vars: bool = False,
        pip_freeze: bool = False,
        output_path_prefix: str | None = None,
        schedule: Schedule | None = None,
    ) -> None:
        """Introspection PyTorch Lightning callback.
        This callback helps capture internal model and training environment states
        to help investigate model performance and other training regressions.
        It publishes reports that help compare the internal state between different
        steps and runs.

        Args:
            checksums (set[str] | None): Checksums to collect.
                Use any combination of the following options:
                ``"params"`` - capture checksum for every model parameter,
                ``"buffers"`` - capture checksum for every model buffer (auxiliary tensor),
                ``"grads"`` - capture checksum for every model parameter's gradient,
                ``"rngs"`` - capture checksum for every Random Number Generator (RNG) state,
                ``"optimizers"`` - capture checksum for optimizers state,
                ``"batch"`` - capture checksum for batch data (model input).
                Defaults to no checksum collection.
            tensor_stats (set[str] | None): Tensor details to collect next to checksums,
                Use any combination of the following options:
                ``"shape"`` - tensor shape,
                ``"dtype"`` - tensor dtype,
                ``"infs"`` - count the number of positive and negative infinity elements (∞̟ and ∞̠),
                ``"nans"`` - count the number of NaN elements (⚠),
                ``"zeros"`` - count the number of 0 elements (⌀),
                ``"med"`` - median value (m̃),
                ``"mean"`` - mean value (μ),
                ``"amean"`` - absolute mean value (μ⁺),
                ``"std"`` - standard deviation (σ),
                ``"var"`` - variance (biased) (σ²),
                ``"uvar"`` - unbiased variance (s²),
                ``"skew"`` - skewness (γ₁),
                ``"kurt"`` - kurtosis (γ₂),
                ``"mode"`` - mode (Mo),
                ``"min"`` - min value (↤),
                ``"max"`` - max value (↦),
                ``"hash"`` - tensor content hash,
                Defaults to ``{"hash"}``
            env_vars (bool): capture environment variables when the training starts,
                defaults to ``False``
            pip_freeze (bool): capture installed pip packages when the training starts,
                defaults to ``False``
            output_path_prefix (str | None): output path prefix for generated reports,
                use to persist these files locally, defaults to temporary location
                that is cleaned as soon as the published by logger.
            schedule (Schedule | None): Controls when logging occurs during training.
                Defaults to :class:`Never`.
        """
        self.checksums = set(checksums or set())
        self.tensor_stats = set(tensor_stats or {"hash"})
        self.env_vars = env_vars
        self.pip_freeze = pip_freeze
        self.output_path_prefix = output_path_prefix
        self.schedule = schedule or Never()
        self._should_publish = False
        self._cb_logger: LightningLogger | None = None
        self.checksum: Any | None = None
        self.grad_hooks: list[Any] = []
        self.gradients: dict[str, tuple[Any, str]] = {}
        self._hooks_registered = False

    def _publish(self, file_name: str, path: str, data: dict[str, Any]) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_file = os.path.join(self.output_path_prefix or td, file_name)
            with open(output_file, "w", encoding="utf-8") as f:
                yaml.dump(data, f, sort_keys=False, indent=2, default_flow_style=False, allow_unicode=True, width=10**6)
            self._cb_logger.log_artifact(output_file, path)  # type: ignore[union-attr]

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        self._cb_logger = CallbackLogger(trainer)
        self.stage = stage
        if self.env_vars:
            self._publish("env_vars.yaml", PATH_PREFIX, dict(os.environ))
        if self.pip_freeze:
            self._publish(
                "pip_freeze.yaml",
                PATH_PREFIX,
                {p.metadata["Name"]: p.version for p in distributions()},
            )

    @override
    def on_train_start(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Register hooks at the start of training to capture all gradients including the first step"""
        self._register_hooks(pl_module)

    @override
    def on_train_end(self, trainer: "L.Trainer", pl_module: "L.LightningModule") -> None:
        """Remove hooks at the end of training"""
        self._remove_hooks()

    def _remove_hooks(self) -> None:
        """Remove all registered hooks"""
        for hook in self.grad_hooks:
            hook.remove()
        self.grad_hooks = []
        self._hooks_registered = False

    def _register_hooks(self, pl_module: "L.LightningModule") -> None:
        """Register hooks on parameters to capture gradients"""
        if self._hooks_registered or "grads" not in self.checksums:
            return
        for name, param in pl_module.named_parameters():
            if param.requires_grad:

                def hook(param_name: str, grad: torch.Tensor) -> torch.Tensor:
                    if grad is not None:
                        g = grad.detach()
                        grad_cks = tensor_checksum(g)
                        self.gradients[param_name] = (grad_cks, _format(g, grad_cks.hexdigest(), self.tensor_stats))
                    return grad

                hook = param.register_hook(partial(hook, name))
                self.grad_hooks.append(hook)
        self._hooks_registered = True

    @override
    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        if self.schedule.check(stage="train", batch_idx=batch_idx, step=trainer.global_step, trainer=trainer):
            self.checksum = {}

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.gradients.clear()
        if self.checksum is None:
            return
        if "batch" in self.checksums:
            self.checksum["batch"] = _batch_checksums(batch, batch_idx, self.tensor_stats)
        self._publish(
            f"rank{trainer.global_rank}.yaml",
            f"{PATH_PREFIX}/{self.stage}/step={trainer.global_step}",
            self.checksum,
        )
        self.checksum = None

    @override
    def on_before_optimizer_step(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.checksum is None:
            return
        self.checksum.update(
            get_checksums(
                trainer,
                pl_module,
                self.gradients,
                self.checksums,
                self.tensor_stats,
            )
        )
