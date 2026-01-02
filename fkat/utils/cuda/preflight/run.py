# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import time
import socket
import sys
import subprocess
import torch
import mlflow
import torch.multiprocessing as mp
import logging
from typing import Any
from fkat.utils.cuda.preflight.health_check.gpu_stress_test import run_gpu_stress_test
from fkat.utils.cuda.preflight.health_check.gpu_connection_test import run_gpu_connection_test
from fkat.utils.aws.imds import instance_metadata
from fkat.utils.cuda.preflight.health_check.helpers import (
    fetch_gpu_info,
    generate_test_folder_name,
    InstanceStats,
    strip_aws_batch_id,
    checkfunction_timeout_manager,
    UniqueID,
    destroy_process_group_if_initialized,
)
from fkat.utils.cuda.preflight.health_check.logger import (
    end_all_mlflow_active_runs,
    initialize_mlflow,
)
from fkat.utils.cuda.preflight.health_check.aws_instance_config import INSTANCE_BENCHMARK_CONFIGS
from fkat.utils.cuda.preflight.health_check.ddb_client import HealthStatusDDBClient
from fkat.utils.cuda.preflight.health_check.constants import (
    STRESS_TEST_MAX_RUNTIME_IN_SEC,
    NUM_LOOPS_RANGE,
    SINGLE_NODE_LATENCY_THRESHOLD_FACTOR,
    PAIR_NODES_LATENCY_THRESHOLD_FACTOR,
    AWS_BATCH_JOB_ID,
    PREFLIGHT_MLFLOW_METRIC_PREFIX,
    HEALTH_CHECK_TIMEOUT_SECS,
)

PG_TIMEOUT_MIN = float(os.environ.get("PG_TIMEOUT_MIN", 2))
MLFLOW_CHECK_INTERVAL_SECS = 5  # check mlflow every 5 second

test_folder_name = generate_test_folder_name()

logger = logging.Logger(__name__)


def _is_result_within_threshold(result: float, baseline: float, factor: float) -> bool:
    """
    Determines if a test result is within an acceptable threshold.

    Args:
        result (float): The measured test result.
        baseline (float): The baseline or expected result value.
        factor (float): The multiplicative threshold factor.

    Returns:
        bool: True if the result is within (baseline * factor), False otherwise.
    """
    threshold = baseline * factor
    return result <= threshold


def _run_single_gpu_stress_test(
    unique_id: UniqueID,
    instance_stats: InstanceStats,
) -> bool:
    """
    Executes a single GPU stress test to evaluate basic GPU performance.

    If the current process is local_rank == 0, it runs the stress test and logs results to MLflow.
    Other ranks wait for MLflow to log the result.

    Args:
        unique_id (UniqueID): Metadata of the current process containing rank and cluster topology.
        instance_stats (InstanceStats): System-specific configuration and GPU info.

    Returns:
        bool: True if the test passes all conditions, False otherwise.
    """

    instance_type = instance_stats.instance_type

    result = {}
    is_passed = True

    if unique_id.local_rank == 0:
        logger.info("\n\n************** Start run_gpu_stress_test() ***********")
        hostname = socket.gethostname()

        try:
            result = checkfunction_timeout_manager(
                run_gpu_stress_test,
                kwargs={
                    "gpu_mem": INSTANCE_BENCHMARK_CONFIGS[instance_type].gpu_memory_gb,
                    "max_runtime": STRESS_TEST_MAX_RUNTIME_IN_SEC,
                },
            )
            baseline_num_loops_low_limit = (
                INSTANCE_BENCHMARK_CONFIGS[instance_type].baseline_num_loops - NUM_LOOPS_RANGE
            )
            baseline_num_loops_up_limit = INSTANCE_BENCHMARK_CONFIGS[instance_type].baseline_num_loops + NUM_LOOPS_RANGE
            is_within_loops_range = (
                baseline_num_loops_low_limit <= int(result["num_loop"]) <= baseline_num_loops_up_limit
            )

            is_passed = is_within_loops_range and result.get("all_gpus", False)
            logger.info(f"{hostname}: {result}")
        except Exception as e:
            is_passed = False
            logger.error(f"Single GPU stree test meets error: {str(e)}")
            result = {"error": str(e)}
        finally:
            torch.cuda.empty_cache()

        mlflow.log_metric(f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/gpu_stress_test/is_test_pass", is_passed)  # type: ignore[possibly-unbound-attribute]

        if "error" not in result:
            logger.info("Update instance stats to include test result")
            instance_stats.gpu_info = {
                str(instance_stats.gpu_info[key]["uuid"]): result[str(key)]
                for key in instance_stats.gpu_info  # type: ignore[index]
            }

        logger.info("\n\n************** Finish run_gpu_stress_test() ***********")
    else:
        max_retries = HEALTH_CHECK_TIMEOUT_SECS // MLFLOW_CHECK_INTERVAL_SECS + 1
        attempt = 0
        metric_name = f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/gpu_stress_test/is_test_pass"
        while attempt < max_retries:
            active_run = mlflow.active_run()  # type: ignore[possibly-unbound-attribute]
            if not active_run:
                raise Exception("Can't find mlflow run.")

            run_data = mlflow.get_run(active_run.info.run_id).data  # type: ignore[possibly-unbound-attribute]
            if metric_name not in run_data.metrics:
                attempt += 1
                time.sleep(MLFLOW_CHECK_INTERVAL_SECS)
            else:
                break  # metric was found, exit loop

        logger.info("preflight on local_rank check ends")

    return is_passed


def _run_single_node_nvlink_test(
    unique_id: UniqueID,
    instance_stats: InstanceStats,
) -> bool:
    """
    Runs a single-node NVLink bandwidth test to validate GPU interconnects within a node.

    Args:
        unique_id (UniqueID): Metadata of the current process containing rank and cluster topology.
        instance_stats (InstanceStats): Configuration and GPU data for this node.

    Returns:
        bool: True if the NVLink test latency is within acceptable limits, False otherwise.
    """
    instance_type = instance_stats.instance_type
    logger.info(f"\n\n************** Start nvlink_test() for unique_id {unique_id} ***********")

    results = {}

    try:
        print(
            f"start single node test, with master_addr {str(unique_id.master_addr)}, "
            f"world_size {unique_id.gpu_per_node}, "
            f"rank {unique_id.local_rank}, local_rank {os.environ['LOCAL_RANK']}"
        )
        results = checkfunction_timeout_manager(
            func=run_gpu_connection_test,
            kwargs={
                "dim_items": int(INSTANCE_BENCHMARK_CONFIGS[instance_type].config_dim ** 0.5),
                "loops": int(INSTANCE_BENCHMARK_CONFIGS[instance_type].num_loops_single),
                "rail": unique_id.local_rank,
                "mode": "single",
                "master_addr": str(unique_id.master_addr),
                "master_port": os.environ["MASTER_PORT"],
                "world_size": unique_id.gpu_per_node,
                "rank": unique_id.local_rank,
            },
        )
        is_passed = _is_result_within_threshold(
            result=results[-1],
            baseline=INSTANCE_BENCHMARK_CONFIGS[instance_type].baseline_single_node_latency,
            factor=SINGLE_NODE_LATENCY_THRESHOLD_FACTOR,
        )

    except Exception as e:
        logger.error(f"Single node nvlink test failed with error: {str(e)}")
        is_passed = False
        results = {"error": str(e)}
    finally:
        destroy_process_group_if_initialized()

    mlflow.log_metric(  # type: ignore[possibly-unbound-attribute]
        f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/single_node_test/is_test_pass_gpu_{unique_id.local_rank}",
        is_passed,
    )

    logger.info(
        f"\n\n************** Finish nvlink_test() for [unique_id, node_rank] is [{unique_id.__dict__}], "
        f"results is [{results}] ***********"
    )

    return is_passed


def _run_multi_nodes_nvlink_test(
    job_level_mlflow_run_id: str,
    unique_id: UniqueID,
    test_nodes_size: int,
    instance_stats: InstanceStats,
) -> bool:
    """
    Performs a multi-node NVLink test across two or more nodes to verify inter-node GPU connectivity.

    Args:
        job_level_mlflow_run_id (str): MLflow run ID for job-level coordination.
        unique_id (UniqueID): Metadata of the current process containing rank and cluster topology.
        test_nodes_size (int): Number of nodes involved in each test group (typically 2).
        instance_stats (InstanceStats): GPU and instance-level metadata.

    Returns:
        bool: True if latency is within expected bounds, False otherwise.
    """

    instance_type = instance_stats.instance_type
    node_pair_rank = unique_id.rank // test_nodes_size
    node_pair_loc_rank = unique_id.rank % (test_nodes_size * unique_id.gpu_per_node)

    job_level_mlflow_run_params = mlflow.get_run(job_level_mlflow_run_id).data.params  # type: ignore[possibly-unbound-attribute]

    logger.info(f"node_rank is {unique_id.node_rank} and node_pair_id is {node_pair_rank}")
    logger.info(
        f"\n\n************** Start {test_nodes_size}-node nvlink_test() for node {unique_id.node_rank} ***********"
    )

    # "MASTER_ADDR" of pg should use "node 0"'s address of this node pair, every 2 nodes would join the same pg.
    new_master_addr = job_level_mlflow_run_params.get(
        f"instance_addr_node_{unique_id.node_rank - unique_id.node_rank % 2}"
    )

    results = {}

    try:
        results = checkfunction_timeout_manager(
            func=run_gpu_connection_test,
            kwargs={
                "dim_items": int(INSTANCE_BENCHMARK_CONFIGS[instance_type].config_dim ** 0.5),
                "loops": int(INSTANCE_BENCHMARK_CONFIGS[instance_type].num_loops_multi),
                "rail": int(os.environ["LOCAL_RANK"]),
                "mode": "multi",
                "master_addr": new_master_addr,
                "master_port": os.environ["MASTER_PORT"],
                "world_size": test_nodes_size * unique_id.gpu_per_node,
                "rank": node_pair_loc_rank,
            },
        )
        is_passed = _is_result_within_threshold(
            result=results[-1],
            baseline=INSTANCE_BENCHMARK_CONFIGS[instance_type].baseline_pair_nodes_latency,
            factor=PAIR_NODES_LATENCY_THRESHOLD_FACTOR,
        )

    except Exception as e:
        logger.error(f"Multi node nvlink test failed with error: {str(e)}")
        is_passed = False
        results = {"error": str(e)}
    finally:
        destroy_process_group_if_initialized()

    mlflow.log_metric(  # type: ignore[possibly-unbound-attribute]
        f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/multi_node_test/is_test_pass_gpu_{os.environ['LOCAL_RANK']}",
        is_passed,
    )

    logger.info(
        f"\n\n************** Finish {test_nodes_size}-node nvlink_test() for [unique_id, node_rank, node_pair_id] is"
        f"[{os.environ['LOCAL_RANK']}, {unique_id.node_rank}, {node_pair_rank}], result is [{results}] ***********"
    )

    return is_passed


def _get_upload_instance_info() -> tuple[InstanceStats, str]:
    """
    Retrieves instance metadata and GPU hash ID for the current node.

    Returns:
        tuple[InstanceStats, str]: A tuple containing instance statistics and GPU hash ID.
    """
    gpu_info, instance_gpu_hash_id = fetch_gpu_info()
    instance_info = instance_metadata()
    instance_stats = InstanceStats(instance_info, gpu_info)

    return instance_stats, instance_gpu_hash_id


def fetch_node_info() -> tuple[bool | str, UniqueID, InstanceStats, str]:
    """
    Gathers necessary metadata for preflight health checking.

    Returns:
        tuple: A tuple containing:
            - fetch success status (bool or error message),
            - UniqueID object,
            - InstanceStats object,
            - Job-level MLflow run ID.
    """
    try:
        instance_stats, instance_gpu_hash_id = _get_upload_instance_info()

        unique_id = UniqueID(
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            local_rank=int(os.environ["LOCAL_RANK"]),
            master_addr=socket.gethostbyname(socket.gethostname()),
            num_nodes=int(os.environ["GROUP_WORLD_SIZE"]),
            gpu_per_node=int(os.environ["LOCAL_WORLD_SIZE"]),
            gpu_hash_id=instance_gpu_hash_id,
            node_rank=int(os.environ["RANK"]) // int(os.environ["LOCAL_WORLD_SIZE"]),
        )

        job_level_mlflow_run_id = initialize_mlflow(unique_id, instance_stats=instance_stats)

        is_passed = True

        logger.info("Successfully get all information about the instance.")
    except Exception as e:
        is_passed = False
        logger.error(f"Failed to get all information about the instance. Error: {str(e)}")

    return is_passed, unique_id, instance_stats, job_level_mlflow_run_id


def _log_result_to_mlflow(instance_stats: InstanceStats) -> None:
    """
    Logs the final instance health status and artifacts to MLflow.

    Args:
        instance_stats (InstanceStats): Instance data including the computed health status.
    """
    mlflow.log_metric(  # type: ignore[possibly-unbound-attribute]
        f"{PREFLIGHT_MLFLOW_METRIC_PREFIX}/isinstance_healthy",
        instance_stats.healthy,
    )

    # Enter the job_level_run mlflow_run
    mlflow.end_run()  # type: ignore[possibly-unbound-attribute]
    mlflow.log_artifacts(test_folder_name)  # type: ignore[possibly-unbound-attribute]


def _log_result_to_ddb(all_check_result: dict[str, Any], unique_id: UniqueID, instance_stats: InstanceStats) -> None:
    """
    Writes the instance's health status and test results to DynamoDB.

    Args:
        all_check_result (dict): Dictionary containing the results of each health check.
        unique_id (UniqueID): Metadata about the instance within the cluster.
        instance_stats (InstanceStats): GPU and instance configuration details.
    """
    try:
        ddb_client = HealthStatusDDBClient(region=instance_stats.instance_region)
        ddb_item = ddb_client.generate_ddb_item(
            instance_gpu_hash_id=unique_id.gpu_hash_id,
            instance_health=instance_stats.healthy,
            gpu_stats=instance_stats.gpu_info,
            batch_job_id=strip_aws_batch_id(os.getenv(AWS_BATCH_JOB_ID, default="local")),
            instance_type=instance_stats.instance_type,
            instance_id=instance_stats.instance_id,
            test_result=all_check_result,
        )
        logger.info(f"Writing {ddb_item} to ddb")
        ddb_client.put_item(ddb_item)
    except Exception as e:
        logger.error(f"Failed to write instance health report to DynamoDB. Error: {str(e)}")


def log_preflight_results(all_check_result: dict[str, Any], unique_id: UniqueID, instance_stats: InstanceStats) -> None:
    """
    Logs the result of the health check to both MLflow and DynamoDB.

    This function only runs on local_rank == 0.

    Args:
        all_check_result (dict): Health check results keyed by test name.
        unique_id (UniqueID): Cluster context and rank information.
        instance_stats (InstanceStats): Node-level configuration and test results.
    """
    # Only write instance health report when local rank is 0
    if unique_id.local_rank == 0:
        # Enter the instance_level mlflow_run
        try:
            _log_result_to_mlflow(instance_stats)

            _log_result_to_ddb(
                all_check_result=all_check_result,
                unique_id=unique_id,
                instance_stats=instance_stats,
            )
        except Exception as e:
            logger.error(f"Failed to write instance health report to mlflow. Error: {str(e)}")


def preflight_health_check() -> None:
    """
    Performs a preflight diagnostic to validate whether the current instance is suitable for distributed training.

    Steps performed:
    1. Gathers instance metadata, GPU hash ID, and cluster information.
    2. Runs a GPU stress test to verify core GPU functionality.
    3. Executes a single-node NVLink test to validate intra-node GPU connectivity.
    4. Conditionally runs a multi-node NVLink test for inter-node GPU connectivity (if node count is even and >1).
    5. Aggregates all test results and determines the node's overall health.
    6. Logs the test results and health status to MLflow and DynamoDB.
    7. Cleans up any distributed process groups and MLflow state.

    Side Effects:
        - Updates the instance health status in MLflow and DynamoDB.
        - Logs diagnostic outputs and results.
        - Delays execution based on rank and test coordination logic.

    Note:
        This function must be called within a properly initialized distributed environment with expected env vars:
        `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `GROUP_WORLD_SIZE`.

    Raises:
        None directly, but will log and mark the instance as unhealthy if any test fails.
    """
    is_all_check_success = True
    all_check_result: dict[str, str | int] = {}

    # fetch instance info
    (
        all_check_result["fetch_instance_info"],
        unique_id,
        instance_stats,
        job_level_mlflow_run_id,
    ) = fetch_node_info()
    logger.info(f"fetching result is {is_all_check_success}")

    # Run gpu stress test
    all_check_result["single_gpu_stress_test"] = all_check_result[
        "fetch_instance_info"
    ] and _run_single_gpu_stress_test(unique_id=unique_id, instance_stats=instance_stats)
    logger.info(
        f"_run_single_gpu_stress_test result is {all_check_result['single_gpu_stress_test']}, "
        f"in local_rank {unique_id.local_rank}"
    )

    # Wait until the port is available
    time.sleep(10)

    # Run single-node test
    all_check_result["single_node_nvlink_test"] = all_check_result[
        "fetch_instance_info"
    ] and _run_single_node_nvlink_test(unique_id=unique_id, instance_stats=instance_stats)
    logger.info(
        f"_run_single_node_nvlink_test result is {all_check_result['single_node_nvlink_test']}, "
        f"in local_rank {unique_id.local_rank}"
    )

    # Wait until the port is available
    time.sleep(10)

    # Run multi-node test
    if unique_id.num_nodes == 1:
        logger.info("Due to nnode is 1, skipping multi_nodes_nvlink_test")
        all_check_result["multi_node_nvlink_test"] = "Not Checked"
    elif unique_id.num_nodes % 2 != 0:
        logger.info("Due to nnode is odd, skipping multi_nodes_nvlink_test for now")
        all_check_result["multi_node_nvlink_test"] = "Not Checked"
    else:
        all_check_result["multi_node_nvlink_test"] = all_check_result[
            "fetch_instance_info"
        ] and _run_multi_nodes_nvlink_test(
            job_level_mlflow_run_id=job_level_mlflow_run_id,
            unique_id=unique_id,
            test_nodes_size=2,
            instance_stats=instance_stats,
        )
    logger.info(
        f"_run_multi_nodes_nvlink_test result is {all_check_result['multi_node_nvlink_test']}, "
        f"in local_rank {unique_id.local_rank}"
    )

    # After we add try...except to catch error, we can log if the instance is unhealthy/healthy.
    instance_stats.healthy = all(all_check_result.values())
    logger.info(
        f"After all the tests local rank : {unique_id.local_rank} has health status of {instance_stats.healthy}"
    )

    log_preflight_results(
        all_check_result=all_check_result,
        unique_id=unique_id,
        instance_stats=instance_stats,
    )

    if not instance_stats.healthy:
        logger.error(f"This instance is not healthy, with test result {all_check_result}")

    end_all_mlflow_active_runs()
    destroy_process_group_if_initialized()

    logger.info("Preflight check ends successfully.")


def isolate_bad_node() -> None:
    """
    Checks the health status of the current instance from DynamoDB and isolates it if unhealthy.

    This function:
    1. Retrieves GPU hash ID and instance metadata.
    2. Queries the health status record from DynamoDB using the GPU hash ID.
    3. If the instance was never scanned, raises an error to indicate unexpected behavior.
    4. If the instance is unhealthy, the process enters an infinite sleep to prevent further participation.
    5. If the instance is healthy, it sleeps for 15 minutes to allow other nodes to complete isolation logic.

    This function is typically used in orchestration flows to quarantine failed nodes.

    Raises:
        RuntimeError: If the instance health record is missing in the database.
    """
    _, instance_gpu_hash_id = fetch_gpu_info()
    instance_meta = instance_metadata()

    ddb_client = HealthStatusDDBClient(instance_meta.region)

    instance_stats = ddb_client.get_item(partition_key=instance_gpu_hash_id)

    if not instance_stats:
        raise RuntimeError(
            f"Instance with instance_id {instance_meta.instance_id}, gpu_hash_id {instance_gpu_hash_id},"
            "has not been scaned."
        )
    elif instance_stats and not instance_stats["healthy"]:
        logger.info("This node is unhealthy, sleep this node forever.")
        while True:
            time.sleep(1)
    else:
        logger.info("This node is healthy, Sleep for 15 min to wait for other isolation job finish.")
        time.sleep(15 * 60)


def check() -> None:
    """
    Executes the current script using the system Python interpreter.

    Intended as a CLI entry point for basic validation or debugging.
    """
    subprocess.run([sys.executable, __file__])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    preflight_health_check()
