from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .disk_numpy_array import DiskNumpyArray


class DiskNumpyArrayIterable(DiskNumpyArray):
    def __init__(self, *args, prefetch=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.prefetch = prefetch
        self.data = None
        self.block_idx = None

    def block_generator(self):
        if isinstance(self.indices, tuple):
            for block_idx in range(
                self.get_block(self.indices[0])[0],
                self.get_block(self.indices[1])[0] + 1,
            ):
                yield block_idx
        elif isinstance(self.indices, list):
            last_block = None
            for i in self.indices:
                block_idx, _ = self.get_block(i)
                if last_block is None or block_idx != last_block:
                    yield block_idx
                    last_block = block_idx

    def prefetch_block(self, block_generator, executor):
        try:
            block_idx = next(block_generator)
        except StopIteration:
            return None
        return {
            "block_id": block_idx,
            "task": executor.submit(
                self.load_block,
                self.data_root,
                block_idx,
                self.block_process_func,
            ),
        }

    def __iter__(self):
        self.data = None
        self.block_idx = None
        executor = ThreadPoolExecutor(self.prefetch)
        prefetch_tasks = deque(maxlen=self.prefetch)
        block_generator = self.block_generator()
        while len(prefetch_tasks) < prefetch_tasks.maxlen:
            block = self.prefetch_block(block_generator, executor)
            if block is None:
                break
            prefetch_tasks.append(block)

        for ind in (
            range(*self.indices) if isinstance(self.indices, tuple) else self.indices
        ):
            block_idx, sample_idx = self.get_block(ind)
            if block_idx != self.block_idx:
                data = prefetch_tasks.popleft()
                assert data["block_id"] == block_idx
                self.data = data["task"].result()
                self.block_idx = block_idx

                block = self.prefetch_block(block_generator, executor)
                if block is not None:
                    prefetch_tasks.append(block)
            yield self.data[sample_idx]
