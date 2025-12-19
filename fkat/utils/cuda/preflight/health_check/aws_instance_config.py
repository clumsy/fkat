# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass


@dataclass(frozen=True)
class InstanceBenchmarkConfig:
    name: str
    gpu_memory_gb: int
    config_dim: int
    num_loops_single: int
    num_loops_multi: int
    baseline_num_loops: int
    baseline_single_node_latency: float
    baseline_pair_nodes_latency: float


P4D_24XLARGE = InstanceBenchmarkConfig(
    name="p4d.24xlarge",
    gpu_memory_gb=18,
    config_dim=4_000_000_000,
    num_loops_single=200,
    num_loops_multi=200,
    baseline_num_loops=43,
    baseline_single_node_latency=261,
    baseline_pair_nodes_latency=772,
)

P4DE_24XLARGE = InstanceBenchmarkConfig(
    name="p4de.24xlarge",
    gpu_memory_gb=18,
    config_dim=4_000_000_000,
    num_loops_single=200,
    num_loops_multi=200,
    baseline_num_loops=43,
    baseline_single_node_latency=261,
    baseline_pair_nodes_latency=772,
)

P5_48XLARGE = InstanceBenchmarkConfig(
    name="p5.48xlarge",
    gpu_memory_gb=40,
    config_dim=8_000_000_000,
    num_loops_single=200,
    num_loops_multi=200,
    baseline_num_loops=24,
    baseline_single_node_latency=324,
    baseline_pair_nodes_latency=419,
)

P5EN_48XLARGE = InstanceBenchmarkConfig(
    name="p5en.48xlarge",
    gpu_memory_gb=75,
    config_dim=16_700_000_000,
    num_loops_single=100,
    num_loops_multi=100,
    baseline_num_loops=9,
    baseline_single_node_latency=660,
    baseline_pair_nodes_latency=780,
)

DEFAULT_INSTANCE = InstanceBenchmarkConfig(
    name="default",
    gpu_memory_gb=10,
    config_dim=1_000,
    num_loops_single=0,
    num_loops_multi=0,
    baseline_num_loops=0,
    baseline_single_node_latency=0,
    baseline_pair_nodes_latency=0,
)

INSTANCE_BENCHMARK_CONFIGS: dict[str, InstanceBenchmarkConfig] = {
    cfg.name: cfg
    for cfg in [
        P4D_24XLARGE,
        P4DE_24XLARGE,
        P5_48XLARGE,
        P5EN_48XLARGE,
        DEFAULT_INSTANCE,
    ]
}
