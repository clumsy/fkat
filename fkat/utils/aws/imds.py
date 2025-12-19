# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""
EC2 Instance Metadata Service related methods
"""

import logging
import socket
from dataclasses import dataclass
from functools import lru_cache

import requests
from requests import HTTPError

from fkat.utils.logging import rank0_logger

IMDS_URL = "http://169.254.169.254/latest"
IMDS_METADATA_URL = f"{IMDS_URL}/meta-data"
IMDS_V2_TOKEN_URL = f"{IMDS_URL}/api/token"
NULL = "_NULL_"  # sentinel value used to mark null (not-available) values


log: logging.Logger = rank0_logger(__name__)

Token = str


@dataclass
class InstanceMetadata:
    """
    Struct representing the instance metadata as fetched from IMDS on the current host.
    Use :py:func:`fkat.utils.aws.imds.instance_metadata` to get a filled-out
    instance of this object.
    """

    instance_id: str
    instance_type: str
    hostname: str
    public_hostname: str
    local_hostname: str
    local_ipv4: str
    availability_zone: str
    region: str
    ami_id: str


@lru_cache
def fetch(metadata: str = "", token: Token | None = None) -> str | None:
    """
    Fetches the specified ``metadata`` from EC2's Instance MetaData Service (IMDS) running
    on the current host, by sending an HTTP GET request to ``http://169.254.169.254/latest/meta-data/<metadata>``.

    To get a list of all valid values of ``metadata`` run this method with no arguments then split
    the return value by new-line.


    Arguments:
        metadata: Name of the instance metadata to query (e.g. ``instance-type``)
        token: IMDS token

    Returns:
        the specified ``metadata`` or ``None`` if IMDS cannot be reached
    """

    try:
        response = requests.get(f"{IMDS_METADATA_URL}/{metadata}", headers={"X-aws-ec2-metadata-token": token or ""})
    except Exception as e:
        log.warning("Error querying IMDSV2 instance metadata won't be available", exc_info=e)
        return None

    if response.ok:
        return response.text
    else:  # response NOT ok
        try:
            response.raise_for_status()
        except HTTPError:
            return None

    raise AssertionError("Unreachable code!")


def token(timeout: int = 60) -> Token | None:
    """
    Fetches IMDS ``token`` from EC2's Instance MetaData Service (IMDSV2) running
    on the current host, by sending an HTTP GET request to ``http://169.254.169.254/latest/meta-data/<metadata>``.

    Arguments:
        timeout: request timeout

    Returns:
        the specified ``token`` or ``""`` if IMDSV2 cannot be reached
    """

    try:
        response = requests.put(
            IMDS_V2_TOKEN_URL, headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, timeout=timeout
        )
    except Exception as e:
        log.warning("Error querying IMDSV2 instance token won't be available", exc_info=e)
        return ""

    if response.ok:
        return response.text
    else:  # response NOT ok
        error_code_msg = f"IMDS Token response is not ok with status code {response.status_code}"
        log.warning("Error querying IMDSV2 instance token won't be available", exc_info=Exception(error_code_msg))
        return ""

    raise AssertionError("Unreachable code!")


@lru_cache
def instance_metadata() -> InstanceMetadata:
    """
    Fetches IMDS instance metadata for the current host from EC2's Instance Metadata Service (IMDS),
    which typically runs on localhost at ``http://169.254.169.254``.
    If IMDS cannot be reached for any reason returns an instance of :py:class:`InstanceMetadata`
    where all the fields are empty strings.

    .. note::
        This method is memoized (value is cached) hence, only the first call
        will actually hit IMDS, and subsequent calls will return the memoized
        value. Therefore, it is ok to call this function multiple times.

    """
    tkn = token()
    return InstanceMetadata(
        instance_id=fetch("instance-id", tkn) or "localhost",
        instance_type=fetch("instance-type", tkn) or NULL,
        availability_zone=fetch("placement/availability-zone", tkn) or NULL,
        region=fetch("placement/region", tkn) or NULL,
        hostname=fetch("hostname", tkn) or socket.gethostname(),
        local_ipv4=fetch("local-ipv4", tkn) or NULL,
        public_hostname=fetch("public-hostname", tkn) or NULL,
        local_hostname=fetch("local-hostname", tkn) or NULL,
        ami_id=fetch("ami-id", tkn) or NULL,
    )
