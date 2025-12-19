# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import torch
import mlflow
from datetime import timedelta
import torch.distributed as dist
from torch.multiprocessing import Queue

import logging
from fkat.utils.cuda.preflight.health_check.timer import Timer
from fkat.utils.cuda.preflight.health_check.helpers import (
    destroy_process_group_if_initialized,
)
from fkat.utils.cuda.preflight.health_check.logger import end_all_mlflow_active_runs
from fkat.utils.cuda.preflight.health_check.constants import (
    MLFLOW_EXPERIMENT_NAME,
    PREFLIGHT_MLFLOW_METRIC_PREFIX,
)

PG_TIMEOUT_MIN = float(os.environ.get("PG_TIMEOUT", 5))


def run_gpu_connection_test(
    mlflow_run_id: str,
    result_queue: Queue,
    dim_items: int,
    loops: int,
    master_addr: str,
    master_port: str,
    world_size: int,
    rank: int,
    device_id: int | None = None,
    mode: str = "single",
) -> None:
    """
    Runs a GPU connectivity and communication benchmark test using NCCL and logs performance metrics to MLflow.

    This function initializes a distributed process group with NCCL, performs a warm-up `all_reduce`, and
    repeatedly performs `all_reduce` operations to test GPU communication latency. It records timing statistics
    for each iteration, logs them to MLflow, and places the results in the provided queue.

    Args:
        mlflow_run_id (str): The ID of the MLflow run to log metrics under.
        result_queue (Queue): A multiprocessing-safe queue where timing results are pushed.
        dim_items (int): The dimension of the square tensor used for the all_reduce operation.
        loops (int): The number of all_reduce iterations to run for benchmarking.
        master_addr(str): new internal addr of the process group,
        master_port(str): port used for the process group,
        world_size(int): number of processes expected in process_group,
        rank(int): RANK of the current process in the process_group,
        rail (Optional[int], optional): The CUDA device ID to use for testing. If None, defaults to the current device.
        mode (str, optional): Mode label to tag the MLflow metrics. Defaults to "single".

    Returns:
        None
    """
    dist.init_process_group(  # type: ignore[possibly-unbound-attribute]
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=PG_TIMEOUT_MIN),
    )

    if mlflow.active_run():
        end_all_mlflow_active_runs()

    mlflow.set_tracking_uri(uri=os.environ["MLFLOW_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME.format(region=os.environ["AWS_DEFAULT_REGION"]))
    mlflow.start_run(run_id=mlflow_run_id)  # type: ignore[possibly-unbound-attribute]

    logging.info(f"inside run_gpu_connection_test, {dim_items}, {loops}, {device_id}")
    timer = Timer()
    device_id = device_id if device_id is not None else torch.cuda.current_device()

    with timer("cuda"):
        device = torch.device("cuda", device_id)
        torch.cuda.set_device(device_id)
        buffer = torch.ones((dim_items, dim_items), device=device, dtype=torch.float64)

    # warmup
    dist.all_reduce(buffer, op=dist.ReduceOp.AVG, async_op=False)  # type: ignore[possibly-unbound-attribute]

    results = []
    for i in range(loops):
        with timer(f"all_reduce_{i}"):
            with timer("send"):
                waiter = dist.all_reduce(buffer, op=dist.ReduceOp.AVG, async_op=True)  # type: ignore[possibly-unbound-attribute]
            with timer("sync"):
                waiter.wait()
                dist.barrier()  # type: ignore[possibly-unbound-attribute]
            with timer("stat"):
                buffer_sum = buffer.sum().item()  # noqa: F841

        results.append(timer[f"all_reduce_{i}"])

        mlflow.log_metric(  # type: ignore[possibly-unbound-attribute]
            f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/{mode}_node_test/latency_gpu_{os.environ['LOCAL_RANK']}",
            timer[f"all_reduce_{i}"],
        )

    result_queue.put(results)
    destroy_process_group_if_initialized()
