"""Cache DRP bytes on the GPU, with bounded shuffles and identical board splits."""
import time
import numpy as np
import torch
from stream_records import board_hash


class CachedRecords:
    def __init__(self, stream, device):
        self.stream = stream
        self.data = torch.empty((stream.positions, 32), dtype=torch.uint8, device=device)
        self.validation = np.empty(stream.positions, dtype=np.bool_)
        self.offsets = []
        offset = 0
        started = time.monotonic()
        for index, path in enumerate(stream.paths):
            self.offsets.append(offset)
            with open(path, "rb") as source:
                raw = np.frombuffer(source.read(), dtype=np.uint8).reshape(-1, 32).copy()
            count = len(raw)
            self.validation[offset:offset+count] = board_hash(raw) % np.uint64(100) == 0
            tensor = torch.from_numpy(raw)
            if device.type == "cuda":
                tensor = tensor.pin_memory()
            self.data[offset:offset+count].copy_(tensor, non_blocking=False)
            offset += count
            if (index + 1) % 100 == 0 or index + 1 == len(stream.paths):
                print(f"cache {index+1}/{len(stream.paths)} shards, {offset:,} records, {time.monotonic()-started:.1f}s", flush=True)
        assert offset == stream.positions

    def blocks_for_epoch(self, epoch):
        rng = np.random.default_rng(self.stream.seed + epoch)
        for index in rng.permutation(len(self.stream.blocks)):
            file_index, first, count = self.stream.blocks[index]
            offset = self.offsets[file_index] + first
            selected = np.flatnonzero(~self.validation[offset:offset+count])
            rng.shuffle(selected)
            if len(selected):
                indices = torch.from_numpy(selected).to(self.data.device)
                yield self.data[offset:offset+count][indices]
