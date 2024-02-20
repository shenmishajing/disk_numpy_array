import json
import os
from typing import Mapping, Sequence

import numpy as np


def save_meta_data(
    data_root,
    metadata,
    batched_generator=False,
    metadata_file_name="metadata.json",
):
    data = metadata.pop("data")
    if batched_generator:
        data = np.concatenate(data)
    else:
        data = np.stack(data)

    np.save(
        os.path.join(data_root, f"{metadata['block_num']}.npy"),
        data,
    )
    metadata["block_num"] += 1
    metadata["length"] += data.shape[0]
    json.dump(metadata, open(os.path.join(data_root, metadata_file_name), "w"))

    metadata["data"] = []


def recursive_update_metadata(
    data_root,
    block,
    metadata=None,
    batched_generator=False,
    block_size=None,
    block_bytes=16 * 1024**2,
    metadata_file_name="metadata.json",
):
    if metadata is None:
        metadata = {}

    if isinstance(block, np.ndarray):
        if not metadata:
            metadata["length"] = 0
            metadata["block_num"] = 0
            metadata["data"] = []

            if block_size is None:
                metadata["block_size"] = max(block_bytes // block.nbytes, 1)
            else:
                metadata["block_size"] = block_size

            if batched_generator:
                metadata["block_size"] *= block.shape[0]

        metadata["data"].append(block)

        if batched_generator:
            sample_num = sum(len(d) for d in metadata["data"])
        else:
            sample_num = len(metadata["data"])

        if sample_num >= metadata["block_size"]:
            save_meta_data(data_root, metadata, batched_generator, metadata_file_name)
    elif isinstance(block, Mapping):
        for k, v in block.items():
            metadata[k] = recursive_update_metadata(
                os.path.join(data_root, k),
                v,
                metadata.get(k),
                batched_generator,
                block_size,
                block_bytes,
                metadata_file_name,
            )
    elif isinstance(block, Sequence):
        metadata = [
            recursive_update_metadata(
                os.path.join(data_root, f"{i}"),
                b,
                metadata[i]
                if isinstance(metadata, Sequence) and len(metadata) > i
                else None,
                batched_generator,
                block_size,
                block_bytes,
                metadata_file_name,
            )
            for i, b in enumerate(block)
        ]
    return metadata


def recursive_save_metadata(
    data_root, metadata, batched_generator=False, metadata_file_name="metadata.json"
):
    if isinstance(metadata, dict) and "data" in metadata:
        if metadata["data"]:
            save_meta_data(data_root, metadata, batched_generator, metadata_file_name)
    elif isinstance(metadata, Mapping):
        for k, v in metadata.items():
            recursive_save_metadata(
                os.path.join(data_root, k), v, batched_generator, metadata_file_name
            )
    elif isinstance(metadata, Sequence):
        for i, d in enumerate(metadata):
            recursive_save_metadata(
                os.path.join(data_root, f"{i}"),
                d,
                batched_generator,
                metadata_file_name,
            )
