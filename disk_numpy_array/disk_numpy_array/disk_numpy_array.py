import json
import os
from glob import iglob

import numpy as np

from ..utils import recursive_save_metadata, recursive_update_metadata


class DiskNumpyArray:
    def __init__(
        self,
        data_root,
        metadate="metadata.json",
        sample_process_func=None,
        block_process_func=None,
    ):
        self.data_root = data_root

        if isinstance(metadate, str):
            metadate = json.load(open(os.path.join(data_root, metadate)))
        assert metadate["block_size"] * metadate["block_num"] >= metadate["length"]
        self.metadata = metadate
        self.sample_process_func = sample_process_func
        self.block_process_func = block_process_func
        self.indices = metadate["length"]

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        if isinstance(value, int):
            assert value <= self.metadata["length"]
            self._indices = (0, value)
        elif isinstance(value, tuple):
            assert value[0] <= value[1] <= self.metadata["length"]
            self._indices = value
        elif isinstance(value, list):
            sorted_value = sorted(value)
            assert sorted_value[0] >= 0 and sorted_value[-1] <= self.metadata["length"]
            self._indices = value
        else:
            raise ValueError("Invalid indices type")

    def get_block(self, idx):
        return divmod(idx, self.metadata["block_size"])

    def set_range_subset(self, num_shared=1, rank=0):
        length = self.__len__()
        if length < num_shared:
            raise ValueError("Shared number is larger than the length of dataset")

        sample_num, res = divmod(length, num_shared)

        if res == 0:
            begin = sample_num * rank
            end = begin + sample_num
        else:
            if rank < res:
                sample_num += 1
                begin = sample_num * rank
                end = begin + sample_num
            else:
                end = sample_num * rank + res
                begin = end - sample_num - 1

        if isinstance(self.indices, tuple):
            self.indices = (self.indices[0] + begin, self.indices[0] + end)
        elif isinstance(self.indices, list):
            self.indices = self.indices[begin:end]

    # def set_block_alternately_subset(self, num_shared=1, rank=0):
    #     if isinstance(self.indices, tuple):
    #         block_indices = (
    #             self.get_block(self.indices[0]),
    #             self.get_block(self.indices[1]),
    #         )
    #         block_num = block_indices[1][0] - block_indices[0][0]
    #         self.indices = [
    #             i for i in range(self.indices[0] + rank, self.indices[1], num_shared)
    #         ]
    #     elif isinstance(self.indices, list):
    #         self.indices = self.indices[rank::num_shared]

    def set_alternately_subset(self, num_shared=1, rank=0):
        length = self.__len__()
        if length < num_shared:
            raise ValueError("Shared number is larger than the length of dataset")

        _, res = divmod(length, num_shared)

        indices = [i for i in range(rank, length, num_shared)]
        if res != 0 and rank >= res:
            indices.append(rank - res)

        if isinstance(self.indices, tuple):
            self.indices = [self.indices[0] + i for i in indices]
        elif isinstance(self.indices, list):
            self.indices = [self.indices[i] for i in indices]

    @staticmethod
    def load_block(data_root, block_idx, block_process_func=None):
        data = np.load(os.path.join(data_root, f"{block_idx}.npy"))
        if block_process_func is not None:
            data = block_process_func(data)
        return data

    def __len__(self):
        if isinstance(self.indices, tuple):
            return self.indices[1] - self.indices[0]
        elif isinstance(self.indices, list):
            return len(self.indices)
        raise ValueError("Invalid indices type")

    @staticmethod
    def save_to_disk(
        generator,
        data_root,
        batched_generator=False,
        block_size=None,
        block_bytes=16 * 1024**2,
        metadata_file_name="metadata.json",
    ):
        data = None

        for block in generator:
            data = recursive_update_metadata(
                data_root,
                block,
                data,
                batched_generator,
                block_size,
                block_bytes,
                metadata_file_name,
            )
        recursive_save_metadata(data_root, data, batched_generator, metadata_file_name)

    @classmethod
    def from_disk(cls, data_root, metadate="metadata.json", *args, **kwargs):
        data = {}
        for file_path in iglob(os.path.join(data_root, "**", metadate)):
            key_path = os.path.relpath(os.path.dirname(file_path), data_root)
            if key_path:
                key_path_list = key_path.split(os.path.sep)
                cur_data = data
                for key in key_path_list[:-1]:
                    try:
                        key = int(key)
                    except ValueError:
                        pass
                    if key not in cur_data:
                        cur_data[key] = {}
                    cur_data = cur_data[key]
                cur_data[key_path_list[-1]] = cls(
                    os.path.join(data_root, key_path),
                    metadate=metadate,
                    *args,
                    **kwargs,
                )
            else:
                data = cls(data_root, metadate=metadate, *args, **kwargs)
                break
        return data
