# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import socket
import time

import mlflow
import torch
import logging
from fkat.utils.cuda.preflight.health_check.constants import (
    AWS_BATCH_JOB_ID,
    AWS_BATCH_LINK,
    MLFLOW_EXPERIMENT_NAME,
)
from fkat.utils.cuda.preflight.health_check.helpers import (
    InstanceStats,
    UniqueID,
    generate_random_string,
    strip_aws_batch_id,
)


def search_join_mlflow_run(run_name: str) -> None:
    """
    Searches for the most recent active MLflow run with the specified run name and joins it.

    This function looks for an active MLflow run matching the given `run_name` within the current region's
    configured experiment. If a match is found, it starts logging to that run. If no run is found,
    it raises a RuntimeError.

    Args:
        run_name (str): The name of the MLflow run to search for.

    Returns:
        str: The MLflow run ID of the matched run.

    Raises:
        RuntimeError: If no active MLflow run with the specified name is found.
    """
    runs = mlflow.search_runs(  # type: ignore[possibly-unbound-attribute]
        experiment_names=[MLFLOW_EXPERIMENT_NAME.format(region=os.environ["AWS_DEFAULT_REGION"])],
        # Output format should be a list.
        output_format="list",
        # Only searching for active run.
        run_view_type=1,
        max_results=1,
        order_by=["start_time DESC"],
        filter_string=f"attributes.run_name = '{run_name}'",
    )

    if runs:
        logging.info(f"Found parent mlflow run: {runs[0].info.run_id}")
        latest_run_id = runs[0].info.run_id
        mlflow.start_run(latest_run_id)  # type: ignore[possibly-unbound-attribute]
    else:
        raise RuntimeError("Can't find parent mlflow runs.")


def create_job_level_mlflow_run(job_level_mlflow_run_name: str, instance_stats: InstanceStats) -> None:
    """Create job level mlflow run, batch_id if batch job, local if local job.
    This will only be create one time in a job, by rank==0. All other processes wait for 5s.
    """
    # global_rank 0 process initialize the run on mlflow, index by aws_batch_id
    if int(os.environ["RANK"]) == 0:
        mlflow.start_run(run_name=job_level_mlflow_run_name)  # type: ignore[possibly-unbound-attribute]

        mlflow.log_param("instance_type", instance_stats.instance_type)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("scan_datetime", instance_stats.scan_datetime)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("region", instance_stats.instance_region)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("batch_job_id", os.getenv(AWS_BATCH_JOB_ID, "local"))  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param(  # type: ignore[possibly-unbound-attribute]
            "batch_job_link",
            (
                AWS_BATCH_LINK.format(
                    region=instance_stats.instance_region,
                    batch_id=os.environ[AWS_BATCH_JOB_ID],
                )
                if AWS_BATCH_JOB_ID in os.environ
                else "local"
            ),
        )
    else:
        time.sleep(5)


def create_instance_level_mlflow_run(
    unique_id: UniqueID,
    job_level_mlflow_run_name: str,
    instance_stats: InstanceStats,
) -> None:
    """
    Creates a job-level MLflow run and logs instance metadata.

    This function should be called once per job (typically by the global rank 0 process).
    It starts an MLflow run with the provided name and logs instance metadata such as type,
    region, and batch job information. All non-zero rank processes will wait for 2 seconds
    to ensure the run is created before proceeding.

    Args:
        unique_id(str): ID of the instance.
        job_level_mlflow_run_name (str): The name to assign to the MLflow run.
        instance_stats (InstanceStats): An object containing metadata about the instance,
                                        including type, region, and scan timestamp.

    Returns:
        None
    """
    # Only the first process of each instance can join the mlflow run
    if os.environ["LOCAL_RANK"] == "0":
        # If global_rank is 0, we don't need to search for it in mlflow.
        logging.info(f"Start the instance_level layer mlflow run in node {instance_stats}.")
        if os.environ["RANK"] != "0":
            search_join_mlflow_run(run_name=job_level_mlflow_run_name)

        mlflow.log_param(f"instance_addr_node_{unique_id.node_rank}", socket.gethostname())  # type: ignore[possibly-unbound-attribute]

        mlflow.start_run(run_name=unique_id.gpu_hash_id, nested=True, log_system_metrics=True)  # type: ignore[possibly-unbound-attribute]
        mlflow.set_system_metrics_node_id(unique_id.node_rank)  # type: ignore[possibly-unbound-attribute]

        mlflow.log_param("instance_id", instance_stats.instance_id)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("instance_gpu_hash_id", unique_id.gpu_hash_id)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param("node_rank", unique_id.node_rank)  # type: ignore[possibly-unbound-attribute]
        mlflow.log_param(  # type: ignore[possibly-unbound-attribute]
            f"gpu_uuid_gpu_{os.environ['LOCAL_RANK']}",
            instance_stats.gpu_info[torch.cuda.current_device()]["uuid"],
        )
        mlflow.log_param(  # type: ignore[possibly-unbound-attribute]
            f"gpu_serial_gpu_{os.environ['LOCAL_RANK']}",
            instance_stats.gpu_info[torch.cuda.current_device()]["serial"],
        )
        mlflow.log_param(f"global_rank_gpu_{os.environ['LOCAL_RANK']}", os.environ["RANK"])  # type: ignore[possibly-unbound-attribute]
        instance_stats.upload_mlflow(unique_id.gpu_hash_id)
    else:
        time.sleep(5)


def get_parent_mlflow_id() -> str:
    """
    Initializes a two-layer MLflow run structure for organized metric and artifact tracking.

    This function sets up the MLflow tracking URI and experiment based on the instance's region.
    It then creates:
    1. A **job-level run** identified by the AWS Batch Job ID (or a local fallback).
    2. An **instance-level run** identified by the instance's GPU hash ID.

    The job-level run is created once by the global rank 0 process. The instance-level run is created
    by local rank 0 processes per node. All other local ranks on a node join the corresponding instance-level run.

    Args:
        node_rank (int): The global rank of the current node (used for job-level run creation).
        instance_gpu_hash_id (str): A unique identifier for the current instance's GPU setup.
        instance_stats (InstanceStats): Object containing instance metadata such as type, region, and scan time.

    Returns:
        str: The MLflow run ID of the job-level (parent) run.
    """
    if active_run := mlflow.active_run():
        current_run_id = active_run.info.run_id
    else:
        raise Exception("mlflow is not activated.")

    if parent_run := mlflow.get_parent_run(current_run_id):  # type: ignore[possibly-unbound-attribute]
        parent_run_id = parent_run.info.run_id
    else:
        raise Exception("instance level mlflow run should have a parent run.")

    return parent_run_id


def initialize_mlflow(unique_id: UniqueID, instance_stats: InstanceStats) -> str:
    """Initial mlflow. The MLflow run will have 2 layers, index by the following:
    1. batch_run_id or "local_********"
    2. Instance_gpu_hash_id.

    In this way metrics/parameter/artifact can be better organized.
    """
    # Set up mlflow endpoint and experiment by region
    logging.info("Initializing MLflow")

    mlflow.set_tracking_uri(uri=os.environ["MLFLOW_URI"])
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME.format(region=instance_stats.instance_region))

    logging.info("Start the job_level layer mlflow run")

    job_level_mlflow_run_name = strip_aws_batch_id(os.getenv(AWS_BATCH_JOB_ID, f"local-{generate_random_string(10)}"))

    # global_rank 0 process initialize the job-level run on mlflow, name by aws_batch_id
    create_job_level_mlflow_run(job_level_mlflow_run_name, instance_stats)

    # local_rank 0 process initialize the instace-level run on mlflow, name by gpu_hash_id
    create_instance_level_mlflow_run(unique_id, job_level_mlflow_run_name, instance_stats)

    # other processes join the instace-level run
    if os.environ["LOCAL_RANK"] != "0":
        search_join_mlflow_run(run_name=unique_id.gpu_hash_id)

    # return the job-level mlflow run id
    return get_parent_mlflow_id()


def end_all_mlflow_active_runs() -> None:
    """End all active mlflow runs."""
    while active_run := mlflow.active_run():
        logging.info(f"Ending run: {active_run.info.run_id}")
        mlflow.end_run()  # type: ignore[possibly-unbound-attribute]
