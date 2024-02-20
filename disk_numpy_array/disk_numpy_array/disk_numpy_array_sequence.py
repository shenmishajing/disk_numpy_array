from sys import getsizeof

from ..replacement_strategy import SampleNumReplacementStrategy
from .disk_numpy_array import DiskNumpyArray


class DiskNumpyArraySequence(DiskNumpyArray):
    def __init__(
        self,
        *args,
        cache_bytes=1 * 1024**3,
        cache_num=None,
        replacement_strategy=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.data = {}
        self.cache_bytes = cache_bytes
        self.cache_num = cache_num

        if replacement_strategy is None:
            replacement_strategy = SampleNumReplacementStrategy()
        self.replacement_strategy = replacement_strategy

    def get_index(self, idx):
        if isinstance(self.indices, tuple):
            return self.indices[0] + idx
        elif isinstance(self.indices, list):
            return self.indices[idx]

    def get_cache_num(self):
        if self.cache_num is None:
            if not self.data:
                raise ValueError("No data loaded")
            block = next(self.data.items())
            self.cache_num = max(self.cache_bytes // getsizeof(block), 1)
        return self.cache_num

    def get_block_to_drop(self):
        if self.data and self.cache_num <= len(self.data):
            return self.replacement_strategy.get_block_to_drop(
                [(k, v["state"]) for k, v in self.data.items()]
            )
        return None

    def __getitem__(self, idx):
        block_idx, sample_idx = self.get_block(self.get_index(idx))
        if block_idx not in self.data:
            block_to_drop = self.get_block_to_drop()
            if block_to_drop is not None:
                del self.data[block_to_drop]

            data = self.load_block(self.data_root, block_idx, self.block_process_func)
            self.data[block_idx] = {
                "data": data,
                "state": self.replacement_strategy.get_init_state(data),
            }
        self.data[block_idx]["state"] = self.replacement_strategy.update_state(
            self.data[block_idx]["state"]
        )
        data = self.data[block_idx]["data"][sample_idx]
        if self.sample_process_func is not None:
            data = self.sample_process_func(data)
        return data
