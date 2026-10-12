import pathlib
import unittest
import numpy as np
import torch
from stream_records import RecordStream
from cache_records import CachedRecords

ROOT = pathlib.Path(__file__).resolve().parent.parent


class CacheTest(unittest.TestCase):
    def test_cached_blocks_match_disk_order_and_validation(self):
        stream = RecordStream([ROOT / "data/lichess-pilot.drp"], block_records=117)
        cached = CachedRecords(stream, torch.device("cpu"))
        for epoch in (1, 3):
            disk = list(stream.blocks_for_epoch(epoch))
            memory = list(cached.blocks_for_epoch(epoch))
            self.assertEqual(len(disk), len(memory))
            for expected, actual in zip(disk, memory):
                np.testing.assert_array_equal(expected, actual.numpy())


if __name__ == "__main__":
    unittest.main()
