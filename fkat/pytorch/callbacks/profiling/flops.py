# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry
from datetime import datetime
from pprint import pformat
from typing import Any, Protocol, TYPE_CHECKING, cast
from typing_extensions import override

import lightning as L

if TYPE_CHECKING:
    from lightning.pytorch.utilities.types import STEP_OUTPUT

from fkat.utils.logging import rank0_logger
from fkat.pytorch.loggers import LightningLogger
from fkat.pytorch.callbacks.loggers import CallbackLogger
from fkat.pytorch.schedule import Schedule, Every

logger = rank0_logger(__name__)


class FlopRecipe(Protocol):
    """FlopRecipe provides an estimation of Floating Point Operation (Flop) during a training batch."""

    def get_batch_flop(self, pl_module: "L.LightningModule") -> int: ...


class GPTModel(FlopRecipe):
    """``GPTModel`` FLOP recipe provides an estimation of Floating Point Operation (Flop)
    during a training batch for MegatronGPTModels.
    """

    @staticmethod
    def _calculate_flop(
        batch_size: int,
        seq_length: int,
        num_layers: int,
        kv_channels: int | None,
        num_attention_heads: int,
        num_query_groups: int,
        hidden_size: int,
        ffn_hidden_size: int,
        group_query_attention: bool,
        swiglu: bool,
        padded_vocab_size: int,
        num_moe_layers: int,
        moe_ffn_hidden_size: int,
        num_experts: int,
        moe_router_topk: int,
        consider_activation_recompute: bool = False,
    ) -> int:
        if kv_channels is None:
            kv_channels = hidden_size // num_attention_heads

        params = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "num_layers": num_layers,
            "kv_channels": kv_channels,
            "num_attention_heads": num_attention_heads,
            "num_query_groups": num_query_groups,
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "group_query_attention": group_query_attention,
            "swiglu": swiglu,
            "padded_vocab_size": padded_vocab_size,
            "num_moe_layers": num_moe_layers,
            "moe_ffn_hidden_size": moe_ffn_hidden_size,
            "num_experts": num_experts,
            "moe_router_topk": moe_router_topk,
            "consider_activation_recompute": consider_activation_recompute,
        }

        logger.debug("Called _calculate_flop with parameters:\n%s", pformat(params))

        # Attention projection size.
        query_projection_size = kv_channels * num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / hidden_size
        # Group Query Attention.
        if not group_query_attention:
            num_query_groups = num_attention_heads
        # MoE.
        num_experts_routed_to = 1 if num_experts is None else moe_router_topk
        gated_linear_multiplier = 3 / 2 if swiglu else 1

        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        #       When the HFU is considered, additional forward pass needs to be factored in.
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        compute_factor = 4 if consider_activation_recompute else 3
        expansion_factor = compute_factor * 2 * 2
        moe_ratio = num_moe_layers / num_layers
        dense_ratio = 1.0 - moe_ratio
        return int(
            expansion_factor
            * batch_size
            * seq_length
            * num_layers
            * hidden_size
            * hidden_size
            * (
                # Attention.
                (
                    (1 + (num_query_groups / num_attention_heads) + (seq_length / hidden_size))
                    * query_projection_to_hidden_size_ratio
                )
                # MLP.
                # Interleave
                + dense_ratio * ((ffn_hidden_size / hidden_size) * gated_linear_multiplier)
                + moe_ratio * ((moe_ffn_hidden_size / hidden_size) * num_experts_routed_to * gated_linear_multiplier)
            )
            +
            # Logits
            6 * batch_size * seq_length * hidden_size * padded_vocab_size
        )

    def get_batch_flop(self, pl_module: "L.LightningModule") -> int:
        """Resolves Floating Pointer Operations (Flop) for the given batch size for GPT models."""
        if hasattr(pl_module, "cfg") and isinstance(pl_module.cfg, dict):
            config = cast(dict[str, Any], pl_module.cfg)

            # reject calculating Flop per batch if key is missing from module config
            _mandatory_config_keys = [
                "global_batch_size",
                "encoder_seq_length",
                "num_layers",
                "hidden_size",
                "num_attention_heads",
                "ffn_hidden_size",
            ]

            for key in _mandatory_config_keys:
                if key not in config:
                    raise KeyError(f"Key {key} not presented in module config {config}")

            return GPTModel._calculate_flop(
                batch_size=config.get("global_batch_size", 1),
                seq_length=config.get("encoder_seq_length", 1024),
                num_layers=config.get("num_layers", 24),
                kv_channels=config.get(
                    "kv_channels", config.get("hidden_size", 4096) // config.get("num_attention_heads", 16)
                ),
                num_attention_heads=config.get("num_attention_heads", 16),
                num_query_groups=config.get("num_query_groups", config.get("num_attention_heads", 16)),
                # Default to num_attention_heads
                hidden_size=config.get("hidden_size", 4096),
                ffn_hidden_size=config.get("ffn_hidden_size", 4 * config.get("hidden_size", 4096)),
                # Typical FFN expansion
                group_query_attention=config.get("group_query_attention", False),
                swiglu=config.get("swiglu", False),
                padded_vocab_size=config.get("padded_vocab_size", config.get("vocab_size", 50257)),
                num_moe_layers=config.get("num_moe_layers", 0),
                moe_ffn_hidden_size=config.get("moe_ffn_hidden_size", 0),
                num_experts=config.get("num_experts", 0),
                moe_router_topk=config.get("moe_router_topk", 1),
                consider_activation_recompute=config.get("consider_activation_recompute", False),
            )

        raise TypeError(f"{self.__class__} does not support calculating flop for {pl_module.__class__}")


class Trace(TorchDispatchMode, FlopRecipe):
    """``Trace`` FLOP recipe is a lightweight counter mode that only counts global flops
    without using ModuleTracker to track module hierarchy.

    Example usage

    .. code-block:: python

        with TraceFlopRecipe() as flop_counter:
            mod.sum().backward()
    """

    def __init__(self) -> None:
        super().__init__()

        self.flop_registry = {
            **flop_registry,
        }

        self._reset_flops_count()

    def _reset_flops_count(self) -> None:
        self.flop_counts = {"Global": 0, "Tracked": 0, "Untracked": 0}

    def get_batch_flop(self, pl_module: "L.LightningModule") -> int:
        return self.flop_counts.get("Global", 0)

    def get_tracked_operations_count(self) -> int | None:
        return self.flop_counts.get("Tracked", 0)

    def get_untracked_operations_count(self) -> int | None:
        return self.flop_counts.get("Untracked", 0)

    def _count_flops(self, func_packet: Any, out: Any, args: tuple[()], kwargs: Any) -> Any:
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out_val=out)
            self.flop_counts["Global"] += flop_count
            self.flop_counts["Tracked"] += 1
        else:
            self.flop_counts["Untracked"] += 1

        return out

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple[()] = (), kwargs: Any = None) -> Any:
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        return self._count_flops(func._overloadpacket, out, args, kwargs)

    def __enter__(self) -> TorchDispatchMode:
        self._reset_flops_count()
        super().__enter__()
        return self

    def __exit__(self, *args: tuple[()] | None) -> None:
        super().__exit__(*args)


class Accelerator:
    def __init__(
        self,
        name: str,
        fp32: int,
        fp16: int | None = None,
        bf16: int | None = None,
        int8: int | None = None,
        int4: int | None = None,
    ) -> None:
        self.name = name
        self.fp32 = fp32
        self.fp16 = fp16 or fp32 * 2
        self.bf16 = bf16 or self.fp16
        self.int8 = int8 or self.fp16 * 2
        self.int4 = int4 or self.int8 * 2

    def flops(self, dtype: torch.dtype) -> int:
        if dtype == torch.float32 and self.fp32:
            return self.fp32
        if dtype == torch.float16 and self.fp16:
            return self.fp16
        if dtype == torch.bfloat16 and self.bf16:
            return self.bf16
        if dtype == torch.int8 and self.int8:
            return self.int8
        raise ValueError(f"No {dtype} flops details for {self.name}")


TFLOPS = 10**12

V100 = Accelerator("V100", fp32=130 * TFLOPS)
H100 = Accelerator("H100", fp32=989 * TFLOPS)
H200 = Accelerator("H200", fp32=989 * TFLOPS)
A100 = Accelerator("A100", fp32=312 * TFLOPS)
A10G = Accelerator("A10G", fp32=35 * TFLOPS)
A10 = Accelerator("A10", fp32=int(31.2 * TFLOPS))
L40S = Accelerator("L40S", fp32=int(91.6 * TFLOPS))


def get_flops(dtype: "torch.dtype", device: "torch.device") -> int:
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        if "V100" in gpu_name:
            return V100.flops(dtype)
        elif "H100" in gpu_name:
            return H100.flops(dtype)
        elif "H200" in gpu_name:
            return H200.flops(dtype)
        elif "A100" in gpu_name:
            return A100.flops(dtype)
        elif "A10G" in gpu_name:
            return A10G.flops(dtype)
        elif "A10" in gpu_name:
            return A10.flops(dtype)
        elif "L40S" in gpu_name:
            return L40S.flops(dtype)

    raise ValueError(f"No flops details for {device} with name {gpu_name}")


def dtype(trainer: L.Trainer) -> "torch.dtype":
    if str(trainer.precision).startswith("32"):
        return torch.float32
    if str(trainer.precision).startswith("16"):
        return torch.float16
    if str(trainer.precision).startswith("bf16"):
        return torch.bfloat16
    raise ValueError(f"Can't infer dtype for {trainer.precision}")


class Flops(L.Callback):
    """
    A PyTorch Lightning callback that measures and logs floating-point operations (FLOPs) and
    Model FLOP Utilization (MFU) during training, validation, testing, and prediction.

    This callback helps to monitor the computational efficiency of models by measuring:
    - Total machine FLOPs available
    - Per-batch FLOPs used by the model
    - Model FLOP Utilization (MFU), i.e., how efficiently the model uses the available compute
    - Batch throughput (batches per second)

    It supports two methods for estimating FLOPs:
    1. Tracing-based estimation via a `Trace` context manager.
    2. Formula-based estimation using a predefined GPTModel FLOP calculator.

    Metrics are logged periodically (or once) to the experiment logger (e.g., MLflow) and include:

    * `mfu`: Model FLOP Utilization (traced)
    * `actual_batches_per_sec`: Measured throughput
    * `max_batches_per_sec`: Theoretical max throughput
    * `batch_flops`: FLOPs used in the current batch
    * `batch_flops_from_formula`: FLOPs estimated via formula (if available)
    * `mfu_from_formula`: MFU based on formula-based estimation
    * `tracked_operations`: Number of FLOPs tracked during tracing
    * `untracked_operations`: Number of operations not accounted for by the tracer

    Args:
        schedule (Optional[Schedule]): Controls when logging occurs during training. Defaults to Every 5 batch.
            - FLOPs are always calculated at least once at the beginning.

    Example:
        >>> trainer = L.Trainer(callbacks=[Flops(log_every_n_batches=10)])
    """

    def __init__(self, schedule: Schedule | None = None, *args: Any, **kwargs: Any) -> None:
        """Measures the total floating-point operations per second and MFU.
        Args:
            schedule (Optional[Schedule]): Controls when logging occurs during training. Defaults to Every 5 batch.
                - FLOPs are always calculated at least once at the beginning.
        Returns:
            None
        """
        self.schedule = schedule or Every(n_batches=5)
        self.kwargs = kwargs
        self.trace_flops_recipe: Trace | None = None
        self.gpt_flops_recipe = GPTModel()  # @TODO(hanfange) - make it more configurable
        self.total_flops = self.batch_idx = 0
        self.mfu_from_formula: float | None = None
        self.batch_flops_from_formula: int | None = None
        self.batch_flops = self.mfu = torch.empty(0)
        self.operations_tracked = self.operations_untracked = torch.empty(0)
        self.start_time: datetime | None = None
        self.is_first_batch = True

        self._timer_active = False
        self._cb_logger: LightningLogger | None = None

    @override
    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        self._cb_logger = CallbackLogger(trainer)
        self.total_flops = get_flops(dtype(trainer), pl_module.device) * trainer.num_nodes * trainer.num_devices
        self.batch_flops = torch.tensor(0, dtype=torch.int64, device=pl_module.device)
        self.operations_tracked = torch.tensor(0, dtype=torch.int64, device=pl_module.device)
        self.operations_untracked = torch.tensor(0, dtype=torch.int64, device=pl_module.device)

    def _should_recalulate_batch_flops(self, trainer: "L.Trainer") -> bool:
        """
        _should_recalulate_batch_flops decides whether to recalculate the value of self.batch_flops.
        Recalcuting batch_flops needs to enter flops counter mode and might cause potential vRAM leakage.

        Only calculate batch flops for the first batch.

        Returns: bool
        """
        return self.is_first_batch

    def _should_report_batch_throughput_and_mfu(self, trainer: "L.Trainer") -> bool:
        """
        _should_report_batch_throughput_and_mfu decides whether to publish batch throughput and mfu metrics to MLFlow

        Only report every n batches.

        Returns: bool
        """
        return self.schedule.check(stage="train", batch_idx=self.batch_idx, step=trainer.global_step, trainer=trainer)

    def _start(self, trainer: "L.Trainer", batch_idx: int) -> None:
        self.batch_idx = batch_idx
        if trainer.sanity_checking:
            return

        if self._should_recalulate_batch_flops(trainer):  # calculate number of flops
            self.trace_flops_recipe = Trace()
            self.trace_flops_recipe.__enter__()

        if self._should_report_batch_throughput_and_mfu(trainer):  # report MFU every n batches
            self.start_time = datetime.now()
            self._timer_active = True

    def _stop(self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any) -> None:
        if trainer.sanity_checking:
            return

        if self._should_recalulate_batch_flops(trainer):  # only calculate number of flops once as it is a constant
            self.trace_flops_recipe.__exit__(None, None, None)  # type: ignore[union-attr]
            flop = self.trace_flops_recipe.get_batch_flop(None)  # type: ignore
            self.batch_flops.fill_(flop)
            self.operations_tracked.fill_(self.trace_flops_recipe.get_tracked_operations_count())  # type: ignore
            self.operations_untracked.fill_(self.trace_flops_recipe.get_untracked_operations_count())  # type: ignore
            trainer.strategy.reduce(self.batch_flops, reduce_op="sum")
            trainer.strategy.reduce(self.operations_tracked, reduce_op="sum")
            trainer.strategy.reduce(self.operations_untracked, reduce_op="sum")

        # main node emits metrics for every n batches
        if (
            trainer.global_rank == 0 and self._should_report_batch_throughput_and_mfu(trainer) and self._timer_active
        ):  # report MFU every n batches
            assert self._cb_logger
            assert self.start_time
            self._timer_active = False

            now = datetime.now()
            actual_batches_per_sec = 1 / (now - self.start_time).total_seconds()

            max_batches_per_sec = self.total_flops / self.batch_flops.item()
            mfu = actual_batches_per_sec / max_batches_per_sec

            metrics = {
                "mfu": mfu,
                "actual_batches_per_sec": actual_batches_per_sec,
                "max_batches_per_sec": max_batches_per_sec,
                "batch_flops": self.batch_flops.item(),
                "total_flops": self.total_flops,
                "batch_flops_tracked_operations": self.operations_tracked.item(),
                "batch_flops_untracked_operations": self.operations_untracked.item(),
            }

            # attempt to calculate batch flop using formula-based approach, rank-zero only
            try:
                self.batch_flops_from_formula = self.gpt_flops_recipe.get_batch_flop(pl_module)
                max_batches_per_sec_from_formula = self.total_flops / self.batch_flops_from_formula
                self.mfu_from_formula = actual_batches_per_sec / max_batches_per_sec_from_formula

                # emit mfu calculated from formula, if applicable
                metrics.update(
                    {
                        "mfu_from_formula": self.mfu_from_formula,
                        "batch_flops_from_formula": self.batch_flops_from_formula,
                    }
                )
            except Exception as e:
                logger.debug(f"Could not calculate FLOP using formula: {e}")

            self._cb_logger.log_batch(metrics=metrics, timestamp=int(now.timestamp() * 1e3), step=trainer.global_step)

        self.is_first_batch = False

    @override
    def on_train_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        """
        Called at the beginning of each training batch.

        This method initiates FLOP tracing and throughput timing for the current batch,
        depending on the logging frequency.

        If conditions are met, it:
        - Begins FLOP tracing using a `Trace` context manager.
        - Records the start time to later compute batch throughput.

        Tracing and logging are skipped during sanity checks.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The model being trained.
            batch (Any): The current batch of data.
            batch_idx (int): Index of the current batch.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._start(trainer, batch_idx)

    @override
    def on_train_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """
        Called at the end of each training batch.

        This method finalizes FLOP tracing and logs performance metrics if applicable.

        If conditions are met, it:
        - Ends FLOP tracing and calculates batch-level FLOPs.
        - Aggregates tracked and untracked operations across devices.
        - Computes Model FLOP Utilization (MFU) based on actual vs. theoretical throughput.
        - Optionally estimates FLOPs using a formula-based approach (`GPTModel`).
        - Logs performance metrics (e.g., MFU, throughput, FLOPs) to the experiment logger.

        Logging is only performed on the global rank 0 process and is skipped during sanity checks.

        Args:
            trainer (Trainer): The PyTorch Lightning trainer instance.
            pl_module (LightningModule): The model being trained.
            outputs (STEP_OUTPUT): The outputs from the training step.
            batch (Any): The current batch of data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self._stop(trainer, pl_module, batch)

    @override
    def on_validation_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when the validation batch begins."""
        self._start(trainer, batch_idx)

    @override
    def on_validation_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the validation batch begins."""
        self._stop(trainer, pl_module, batch)

    @override
    def on_predict_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when the predict batch begins."""
        self._start(trainer, batch_idx)

    @override
    def on_predict_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the predict batch begins."""
        self._stop(trainer, pl_module, batch)

    @override
    def on_test_batch_start(
        self, trainer: "L.Trainer", pl_module: "L.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Called when the test batch begins."""
        self._start(trainer, batch_idx)

    @override
    def on_test_batch_end(
        self,
        trainer: "L.Trainer",
        pl_module: "L.LightningModule",
        outputs: "STEP_OUTPUT",
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Called when the test batch begins."""
        self._stop(trainer, pl_module, batch)
