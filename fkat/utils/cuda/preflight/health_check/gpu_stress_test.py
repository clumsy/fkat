# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import math
import mlflow
import time
from typing import Any
import torch
from torch.multiprocessing import Queue
import logging
from fkat.utils.cuda.preflight.health_check.constants import (
    MLFLOW_EXPERIMENT_NAME,
    PREFLIGHT_MLFLOW_METRIC_PREFIX,
)
from fkat.utils.cuda.preflight.health_check.logger import end_all_mlflow_active_runs


def run_gpu_stress_test(mlflow_run_id: str, result_queue: Queue, gpu_mem: int, max_runtime: int) -> None:
    """
    Performs a multi-GPU stress test by executing repeated matrix multiplications and inter-GPU memory transfers.

    This function:
    - Allocates large tensors on each GPU (assuming 8 GPUs),
    - Performs repeated `matmul` operations to stress GPU compute,
    - Copies results across GPUs to test memory transfer integrity,
    - Verifies data correctness after each transfer,
    - Logs metrics to MLflow regarding correctness and loop iterations,
    - Returns a dictionary summarizing the health of each GPU via the result queue.

    Args:
        mlflow_run_id (str): The MLflow run ID under which metrics are logged.
        result_queue (Queue): A multiprocessing-safe queue to place GPU health results.
        gpu_mem (int): Approximate GPU memory (in GB) to target when allocating stress test tensors.
        max_runtime (int): Maximum runtime (in seconds) to perform the stress test.

    Returns:
        None
    """
    if not torch.cuda.is_available():
        result_queue.put({})
        return
    mlflow.set_tracking_uri(uri=os.environ["MLFLOW_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME.format(region=os.environ["AWS_DEFAULT_REGION"]))
    if mlflow.active_run():
        end_all_mlflow_active_runs()

    mlflow.start_run(run_id=mlflow_run_id)  # type: ignore[possibly-unbound-attribute]

    # Get the array size for a square array that fills 1/4 of memory with 2 byte values
    arr_size = (((gpu_mem / 4) * 10**9) / 2) ** (1 / 2)
    arr_size = int(math.ceil(arr_size))
    num_gpus = torch.cuda.device_count()
    logging.info(f"inside run_load(), num_gpus is: {num_gpus}")
    if num_gpus != 8:
        result_queue.put({})
        return

    gpu_health: dict[str, Any] = {str(idx): "Unknown" for idx in range(num_gpus)}
    gpu_health["check_record"] = []

    Ts = [torch.ones(arr_size, arr_size, dtype=torch.bfloat16, device=f"cuda:{gpu_num}") for gpu_num in range(num_gpus)]
    results = [
        torch.zeros(arr_size, arr_size, dtype=torch.bfloat16, device=f"cuda:{gpu_num}") for gpu_num in range(num_gpus)
    ]
    from_others = [
        torch.zeros(arr_size, arr_size, dtype=torch.bfloat16, device=f"cuda:{gpu_num}") for gpu_num in range(num_gpus)
    ]

    torch.manual_seed(12345)

    start_time = time.time()
    curr_loop_num = 0
    while time.time() - start_time < max_runtime:
        # Matrix multiply into result
        # TODO: record the latency of each matmul
        [torch.matmul(T, T, out=result) for T, result in zip(Ts, results, strict=False)]

        # Move into gpu curr_loop_num away
        for i in range(num_gpus):
            other_gpu = (curr_loop_num % (num_gpus - 1) + i + 1) % num_gpus
            other = from_others[other_gpu]
            original = results[i]
            other[:] = original

        # Check values are correct
        checks = [
            (other == result).sum() == result.numel() for other, result in zip(from_others, results, strict=False)
        ]

        for idx, check in enumerate(checks):
            if not check.item():
                gpu_health[str(idx)] = "Unhealthy"

        gpu_health["check_record"].append([check.item() for check in checks])

        curr_loop_num += 1

        mlflow.log_metric(  # type: ignore[possibly-unbound-attribute]
            f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/gpu_stress_test/all_gpu_calculation_is_correct",
            all(checks),
        )
        mlflow.log_metric(  # type: ignore[possibly-unbound-attribute]
            f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/gpu_stress_test/num_loop_gpu_stress_test",
            curr_loop_num,
        )

    gpu_health["num_loop"] = str(curr_loop_num)

    logging.info(f"Finsihed run_gpu_stress_test. {curr_loop_num} loops ran.")

    if curr_loop_num < num_gpus:
        logging.info(f"Few loops seen, only {curr_loop_num}")
        for idx in range(num_gpus):
            gpu_health[str(idx)] = "Unknown"

    gpu_health["all_gpus"] = all(gpu_health.get(str(i)) == "Healthy" for i in range(num_gpus))

    # Free memory
    del Ts, results, from_others
    torch.cuda.empty_cache()

    result_queue.put(gpu_health)
