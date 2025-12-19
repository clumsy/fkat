# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
import logging
from typing import TYPE_CHECKING, Any
from fkat.utils.cuda.preflight.health_check.constants import INSTANCE_HEALTH_STATUS_DDB_TABLE_NAME
import datetime

if TYPE_CHECKING:
    from types_boto3_dynamodb.service_resource import DynamoDBServiceResource, Table


class HealthStatusDDBClient:
    _PARTITION_KEY = "instance_gpu_hash_id"
    _SORT_KEY = "time_checked"
    _is_initialized = False
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> HealthStatusDDBClient:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, region: str = "us-east-1") -> None:
        if not self._is_initialized:
            session = boto3.Session()
            self.ddb_resource: DynamoDBServiceResource = session.resource("dynamodb", region_name=region)  # type: ignore[assignment]
            self.table: Table = self.ddb_resource.Table(INSTANCE_HEALTH_STATUS_DDB_TABLE_NAME)

            self._is_initialized = True

    def generate_ddb_item(
        self,
        instance_gpu_hash_id: str,
        instance_health: bool,
        gpu_stats: dict[str | int, dict[str, Any]],
        batch_job_id: str,
        instance_id: str,
        instance_type: str,
        test_result: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "instance_gpu_hash_id": instance_gpu_hash_id,
            "time_checked": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "batch_job_id": batch_job_id,
            "instance_type": instance_type,
            "gpu_info": {str(key): value for key, value in gpu_stats.items()},
            "instance_id": instance_id,
            "healthy": instance_health,
            "test_result": test_result,
        }

    def put_item(self, item: dict[str, Any]) -> None:
        try:
            respones = self.table.put_item(Item=item, ReturnValues="ALL_OLD")
            logging.info(f"Item {respones} successfully added to ddb.")
        except ClientError as e:
            # If the item already exists, an exception will be raised
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logging.error("Item already exists. Duplicate insertion prevented.")
            else:
                logging.error(f"Error inserting item: {e.response['Error']['Message']}")

            raise e

    def get_item(self, partition_key: str) -> dict[str, Any] | None:
        try:
            response = self.table.query(
                KeyConditionExpression=Key(self._PARTITION_KEY).eq(partition_key),
                ScanIndexForward=False,
                Limit=1,
            )

            logging.info(f"successfully get response {response} from table with key {partition_key}")
            return response["Items"][0] if "Items" in response and response["Items"] else None
        except Exception as e:
            logging.error("An unexpected error occurred:", e)
            raise e
