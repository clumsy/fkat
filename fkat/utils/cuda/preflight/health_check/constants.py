# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
INSTANCE_HEALTH_STATUS_DDB_TABLE_NAME = "NodeHealthStatus"

# GPU stress test configs:
STRESS_TEST_MAX_RUNTIME_IN_SEC = 120
HEALTH_CHECK_TIMEOUT_SECS = 150

# Good/bad node decision factor
NUM_LOOPS_RANGE = 1
SINGLE_NODE_LATENCY_THRESHOLD_FACTOR = 1.1
PAIR_NODES_LATENCY_THRESHOLD_FACTOR = 1.1
PREFLIGHT_MLFLOW_METRIC_PREFIX = "preflight"

# Max len of a node IPv4 address
MAX_ADDR_LENGTH = 100
PASS = "pass"
FAIL = "fail"

MLFLOW_EXPERIMENT_NAME = "bad_node_detection_{region}"

AWS_BATCH_JOB_ID = "AWS_BATCH_JOB_ID"
AWS_BATCH_LINK = "https://{region}.console.aws.amazon.com/batch/home?region={region}#jobs/ec2/detail/{batch_id}"
