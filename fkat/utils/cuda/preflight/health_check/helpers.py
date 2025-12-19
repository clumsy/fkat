# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import os
import random
import string
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

import mlflow
import requests
from requests.exceptions import RequestException
import torch
from dataclasses import dataclass
import time
import logging
from fkat.utils.aws.imds import InstanceMetadata
from fkat.utils.cuda.preflight.health_check.constants import (
    AWS_BATCH_JOB_ID,
    HEALTH_CHECK_TIMEOUT_SECS,
)
from pynvml import (
    NVMLError,
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetSerial,
    nvmlDeviceGetUUID,
    nvmlInit,
    nvmlShutdown,
)
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.multiprocessing import Queue

torch.manual_seed(12345)


@dataclass
class UniqueID:
    rank: int
    world_size: int
    local_rank: int
    num_nodes: int
    node_rank: int
    gpu_per_node: int
    gpu_hash_id: str
    master_addr: str


class InstanceStats:
    def __init__(self, instance_metadata: InstanceMetadata, gpu_info: dict[str | int, dict[str, Any]]) -> None:
        self.instance_id = instance_metadata.instance_id
        self.instance_type = instance_metadata.instance_type
        self.instance_ipv4 = instance_metadata.local_ipv4
        self.instance_hostname = instance_metadata.hostname
        self.instance_region = instance_metadata.region
        self.scan_datetime = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.gpu_info = gpu_info
        self.healthy = True

    def upload_mlflow(self, instance_gpu_hash_id: str) -> None:
        """Upload instance stats to mlflow.

        Args:
            instance_gpu_hash_id (str): hash the sorted instance GPU UUIDs
        """
        mlflow.log_param("instance_gpu_hash_id", instance_gpu_hash_id)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("instance_id", self.instance_id)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("instance_type", self.instance_type)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("instance_ipv4", self.instance_ipv4)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("instance_hostname", self.instance_hostname)  # type: ignore[possibly-unbound-attribute]

        # mlflow param has 300 limit, seperate the gpu_info to serial_number and uuid
        mlflow.log_param("gpu_uuid", {k: v["uuid"] for k, v in self.gpu_info.items()})  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("gpu_serial_number", {k: v["serial"] for k, v in self.gpu_info.items()})  # type: ignore[possibly-unbound-attribute]


def make_requests(url: str, token: str) -> str:
    """Retrieve instance metadata from AWS EC2 Instance Metadata Service (IMDSv2).

    Args:
        url (str): The URL endpoint of the IMDSv2 metadata service.
        token (str): The authentication token required for IMDSv2 requests.

    Returns:
        str: The retrieved instance metadata as a string.

    Raises:
        requests.exceptions.RequestException: If the request fails due to network
        issues, invalid URL, or failed response status.
    """
    try:
        instance_info_response = requests.get(url, headers={"X-aws-ec2-metadata-token": token})
        instance_info_response.raise_for_status()
        instance_info = instance_info_response.text

    except RequestException as e:
        logging.error(f"Failed to retrieve instance metadata: {e}")
        raise e

    return instance_info


def generate_gpu_uuid_hash(uuid_list: list[str]) -> str:
    """
    Concatenates the UUIDs, computes a SHA-256 hash,
    and returns the first 17 hex characters.
    """
    combined_uuid = "".join(sorted(uuid_list))
    sha_hash = hashlib.sha256(combined_uuid.encode("utf-8")).hexdigest()
    return "g-" + sha_hash[:17]  # Take the first 17 hex characters


def fetch_gpu_info() -> tuple[dict[int | str, dict[str, Any]], str]:
    """
    Retrieve GPU information from the current EC2 instance using NVIDIA Management Library (NVML).

    This function initializes NVML to gather GPU details available to PyTorch,
    including GPU UUIDs and Serial Numbers. Additionally, it generates a hash ID
    representing all GPUs' UUIDs for easier identification.

    The function logs relevant information and gracefully handles errors, shutting
    down NVML in all scenarios.

    Returns:
        tuple:
            - gpu_info (dict): A dictionary containing GPU information where keys are
            PyTorch device indices (int) and values are dictionaries with the following keys:
                - 'uuid' (str): The UUID of the GPU.
                - 'serial' (str): The Serial Number of the GPU.

            - instance_gpu_hash_id (str): A hash string representing the combined UUIDs of all GPUs.

    Raises:
        NVMLError: If there's an issue retrieving GPU information from NVML.
    """
    gpu_info: dict[int | str, dict[str, Any]] = {}
    instance_gpu_hash_id = ""
    try:
        nvmlInit()

        # Get number of GPUs available to PyTorch
        num_gpus = torch.cuda.device_count()

        logging.info(f"Total of {num_gpus} exist in the device")

        # Iterate over each GPU by its PyTorch index
        for i in range(num_gpus):
            # Get GPU handle by index
            handle = nvmlDeviceGetHandleByIndex(i)
            # Retrieve the UUID of the GPU
            uuid = nvmlDeviceGetUUID(handle).decode("utf-8")
            serial = nvmlDeviceGetSerial(handle).decode("utf-8")
            logging.info(f"PyTorch Device Index {i} - GPU UUID: {uuid} - Serial Number: {serial}")

            gpu_info[i] = {
                "uuid": uuid,
                "serial": serial,
            }

        instance_gpu_hash_id = generate_gpu_uuid_hash([gpu["uuid"] for gpu in gpu_info.values()])
        logging.info(f"instance_gpu_hash_id is {instance_gpu_hash_id}")

    except NVMLError as e:
        logging.info("error when fetch informatioin of GPU")
        raise e
    finally:
        nvmlShutdown()

    return gpu_info, instance_gpu_hash_id


def generate_random_string(length: int) -> str:
    """
    Generate a random string of specified length containing uppercase letters,
    lowercase letters, and digits.

    Args:
        length (int): The desired length of the generated string.

    Returns:
        str: A randomly generated string of the specified length.
    """
    characters = string.ascii_letters + string.digits  # Uppercase, lowercase letters, and digits
    return "".join(random.choice(characters) for _ in range(length))


def generate_test_folder_name() -> str:
    """
    Generate a unique test folder name using the current timestamp and a random string.

    The folder name is constructed by combining the current date and time (formatted as
    'YYYYMMDD_HHMMSS') with a randomly generated string of 6 characters consisting of
    uppercase letters, lowercase letters, and digits.

    Returns:
        str: A unique test folder name.

    Example:
        >>> generate_test_folder_name()
        '20250324_153045_A3bX7z'
    """
    if AWS_BATCH_JOB_ID in os.environ:
        test_folder_name = strip_aws_batch_id(os.getenv(AWS_BATCH_JOB_ID, "local"))
    else:
        test_folder_name = f"local_{datetime.now(timezone.utc).strftime('%Y-%m-%d-%H-%M-%S')}"
    logging.info(f"current job log will be saved to folder {test_folder_name}")
    return test_folder_name


def strip_aws_batch_id(aws_batch_id: str) -> str:
    """
    Strip the AWS Batch ID to remove any additional node information.

    Args:
        aws_batch_id (str): The original AWS Batch ID, which may include a node index suffix.

    Returns:
        str: The stripped AWS Batch ID without any node index suffix.
    """
    return aws_batch_id.split("#")[0]


def destroy_process_group_if_initialized() -> None:
    """
    Safely destroys the PyTorch distributed process group if it is initialized.

    This function checks if the `torch.distributed` process group is both available
    and initialized. If so, it calls `destroy_process_group()` and logs success.
    Otherwise, it logs a warning. Any exceptions during the process are caught
    and logged as errors.
    """
    try:
        if dist.is_available() and dist.is_initialized():  # type: ignore[possibly-unbound-attribute]
            dist.destroy_process_group()  # type: ignore[possibly-unbound-attribute]
            logging.info("Process group destroyed.")
        else:
            logging.warning("No process group to destroy.")
    except Exception as e:
        logging.error(str(e))
        logging.error("Process group can't be terminated.")


def checkfunction_timeout_manager(func: Callable[..., None], kwargs: dict[str, Any]) -> Any:
    """
    Monitor and enforce a timeout for executing a function within a separate process.

    This function runs a specified function (`func`) in a separate process with
    the provided arguments (`kwargs`). It continuously monitors the execution time
    and terminates the process if it exceeds a defined timeout (`HEALTH_CHECK_TIMEOUT_SECS`).

    The function result is returned via a multiprocessing queue. If the timeout is reached,
    a `TimeoutError` is raised.

    Args:
        func (Callable): The target function to be executed in a separate process.
                         It must accept `mlflow_run_id` and `result_queue` as its first
                         two arguments, followed by additional `kwargs`.
        kwargs (dict): The keyword arguments to be passed to the function being monitored.

    Returns:
        Any: The result returned by the `func` via the multiprocessing queue.
    Raises:
        TimeoutError: If the function exceeds the allowed timeout (`HEALTH_CHECK_TIMEOUT_SECS`).
    """
    active_run = mlflow.active_run()
    if active_run:
        mlflow_run_id = active_run.info.run_id
    else:
        raise Exception("mlflow is not activated.")

    result_queue: Queue[dict[str, Any] | list[int]] = mp.Queue()
    func_process = mp.Process(target=func, args=(mlflow_run_id, result_queue), kwargs=kwargs)
    func_process.start()

    try:
        start_time = time.perf_counter()
        while func_process.is_alive():
            elapsed = time.perf_counter() - start_time
            if elapsed > HEALTH_CHECK_TIMEOUT_SECS:
                logging.error(
                    f"[watchdog] func {func.__name__} Timeout reached ({HEALTH_CHECK_TIMEOUT_SECS} seconds). "  # type: ignore[unresolved-attribute]
                    "Terminating processes."
                )

                # Terminate the processes
                func_process.terminate()

                raise TimeoutError(f"[watchdog] func {func.__name__} exceeded {HEALTH_CHECK_TIMEOUT_SECS:.3f} seconds")  # type: ignore[unresolved-attribute]
            time.sleep(1)

        # If func finished before timeout, clean up the gpu_log process.
        logging.info(f"[watchdog] func {func.__name__} completed in time, used {elapsed}s; terminating gpu_log.")  # type: ignore[unresolved-attribute]
        return result_queue.get()
    finally:
        func_process.terminate()
        func_process.join()
