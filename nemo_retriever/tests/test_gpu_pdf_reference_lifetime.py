# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""GPU-image metadata must outlive the CPU actor that constructed its block."""

import time

import pandas as pd
import pytest
import ray
from ray.data import DataContext
from ray.data.block import BlockAccessor

from nemo_retriever.graph.executor import _preserves_pandas_output
from nemo_retriever.operators.extract.pdf.gpu_transport import GpuPageImage


@ray.remote(num_cpus=0)
class _ImageOwner:
    def publish(self):
        # A CPU value exercises Ray ownership without requiring model weights
        # or GPU hardware. Production uses the same nested ObjectRef contract.
        return {"reference": ray.put([123])}


@ray.remote(num_cpus=0)
class _MetadataEncoder:
    def encode(self, owner):
        reference = ray.get(owner.publish.remote())["reference"]
        frame = pd.DataFrame({"page_image": [{"gpu_image": GpuPageImage(reference, 0, 1, 1)}]})
        context = DataContext.get_current()
        context.batch_to_block_arrow_format = not _preserves_pandas_output(type(self), {"pdf_renderer": "cuda_rdt"})
        return BlockAccessor.batch_to_block(frame)

    def ready(self):
        return True


@ray.remote(num_cpus=0)
class _MetadataConsumer:
    def fetch(self, block):
        assert isinstance(block, pd.DataFrame)
        image = block["page_image"].iloc[0]["gpu_image"]
        return ray.get(image.reference, timeout=15)


def test_image_reference_survives_encoding_actor_exit():
    if ray.is_initialized():
        pytest.skip("This ownership regression requires an isolated local Ray runtime")
    ray.init(address="local", num_cpus=1, num_gpus=0, include_dashboard=False, log_to_driver=False)
    try:
        owner = _ImageOwner.remote()
        encoder = _MetadataEncoder.remote()
        consumer = _MetadataConsumer.remote()
        block = encoder.encode.remote(owner)
        ready, _ = ray.wait([block], timeout=30)
        assert ready
        # Do not deserialize the metadata in the driver before the encoder exits.
        # Otherwise a driver-held nested reference could mask the ownership bug.
        ray.kill(encoder)
        deadline = time.monotonic() + 15
        while True:
            try:
                ray.get(encoder.ready.remote(), timeout=15)
            except ray.exceptions.RayActorError:
                break
            assert time.monotonic() < deadline, "Metadata encoder did not exit"
            time.sleep(0.05)
        assert ray.get(consumer.fetch.remote(block), timeout=30) == [123]
    finally:
        ray.shutdown()
