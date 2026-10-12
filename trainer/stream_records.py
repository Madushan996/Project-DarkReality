"""Bounded-memory DRP blocks and a board-grouped train/validation split.

Never allocate a billion-entry permutation or load every shard into RAM. Board
identity excludes score, result and FEN counters; repeated boards always enter
the same split, including when they occur in separate source shards.
"""
import os
import numpy as np


def board_hash(raw):
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    words = raw[:, :24].copy().view("<u8").reshape(-1, 3)
    with np.errstate(over="ignore"):
        value = words[:, 0] ^ (words[:, 1] * np.uint64(0x9E3779B97F4A7C15))
        value ^= words[:, 2] * np.uint64(0xD6E8FEB86659FD93)
        value ^= raw[:, 27].astype(np.uint64) * np.uint64(0xA0761D6478BD642F)
        value ^= value >> np.uint64(30)
        value *= np.uint64(0xBF58476D1CE4E5B9)
        value ^= value >> np.uint64(27)
        value *= np.uint64(0x94D049BB133111EB)
        value ^= value >> np.uint64(31)
    return value


class RecordStream:
    def __init__(self, paths, block_records=524288, seed=61007):
        self.paths = list(map(str, paths))
        self.block_records = block_records
        self.seed = seed
        self.blocks = []
        self.positions = 0
        for file_index, path in enumerate(self.paths):
            size = os.path.getsize(path)
            if size % 32:
                raise ValueError(f"truncated DRP shard: {path}")
            count = size // 32
            self.positions += count
            self.blocks.extend((file_index, first, min(block_records, count-first))
                               for first in range(0, count, block_records))
        if not self.positions:
            raise ValueError("empty data stream")

    def blocks_for_epoch(self, epoch=0, validation=False):
        rng = np.random.default_rng(self.seed + epoch)
        blocks = rng.permutation(len(self.blocks)) if not validation else range(len(self.blocks))
        for index in blocks:
            file_index, first, count = self.blocks[index]
            # One bounded allocation, closed immediately after reading.
            with open(self.paths[file_index], "rb") as file:
                file.seek(first * 32)
                raw = np.fromfile(file, dtype=np.uint8, count=count*32).reshape(-1, 32)
            is_val = board_hash(raw) % np.uint64(100) == 0
            selected = np.flatnonzero(is_val if validation else ~is_val)
            if not validation:
                rng.shuffle(selected)
            if len(selected):
                yield raw[selected].copy()

    def batches(self, batch_size, epoch=0, validation=False, max_positions=None):
        emitted = 0
        for raw in self.blocks_for_epoch(epoch, validation):
            for start in range(0, len(raw), batch_size):
                batch = raw[start:start+batch_size]
                if max_positions is not None:
                    batch = batch[:max_positions-emitted]
                if not len(batch):
                    return
                yield batch.copy()
                emitted += len(batch)
                if max_positions is not None and emitted >= max_positions:
                    return
