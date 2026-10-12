import pathlib
import tempfile
import unittest
import numpy as np

from stream_records import RecordStream, board_hash

ROOT = pathlib.Path(__file__).resolve().parent.parent


class StreamTest(unittest.TestCase):
    def test_split_coverage_and_grouping_across_shards(self):
        original = np.fromfile(ROOT / "data/lichess-pilot.drp", dtype=np.uint8).reshape(-1, 32)
        altered = original.copy()
        altered[:, 24:27] = 0  # different label/outcome
        altered[:, 28:30] = 9  # different FEN counter
        np.testing.assert_array_equal(board_hash(original), board_hash(altered))
        with tempfile.TemporaryDirectory() as directory:
            paths = [pathlib.Path(directory) / f"shard{i}.drp" for i in range(2)]
            original.tofile(paths[0])
            altered.tofile(paths[1])
            stream = RecordStream(paths, block_records=117)
            train = np.concatenate(list(stream.batches(53, epoch=3)))
            val = np.concatenate(list(stream.batches(53, validation=True)))
            self.assertEqual(len(train) + len(val), len(original) * 2)
            self.assertFalse(set(board_hash(train)) & set(board_hash(val)))
            self.assertEqual(sum(len(batch) for batch in stream.batches(53, validation=True, max_positions=7)), 7)
            self.assertEqual({bytes(row) for row in np.concatenate((train, val))},
                             {bytes(row) for row in np.concatenate((original, altered))})

    def test_truncated_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "broken.drp"
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                RecordStream([path])


if __name__ == "__main__":
    unittest.main()
