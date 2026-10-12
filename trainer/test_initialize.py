import contextlib
import io
import struct
import pathlib
import tempfile
import unittest
import torch
from train import Net, check, decode, fen_to_record
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


class InitializeTest(unittest.TestCase):
    def test_own_network_reexport_and_bucket_replication(self):
        torch.set_num_threads(1)
        original = ROOT / "nets/current.nnue"
        header = original.read_bytes()[:36]
        hidden, buckets = struct.unpack_from("<2I", header, 12)
        king_buckets = struct.unpack_from("<I", header, 20)[0] if header[:8] == b"DRNNUE03" else 1
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            same = pathlib.Path(directory) / "same.nnue"
            net = Net(hidden, buckets=buckets, king_buckets=king_buckets)
            net.initialize_quantized(original)
            net.export(same)
            self.assertEqual(original.read_bytes(), same.read_bytes())
            expanded = pathlib.Path(directory) / "expanded.nnue"
            net = Net(hidden, buckets=8, king_buckets=king_buckets)
            net.initialize_quantized(original)
            net.export(expanded)
            fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                    "8/8/2k5/8/2K5/8/3Q4/8 b - - 0 1"]
            for fen in fens:
                values = []
                for path in (original, expanded):
                    text = io.StringIO()
                    with contextlib.redirect_stdout(text):
                        check(path, fen)
                    values.append(text.getvalue())
                self.assertEqual(*values)

    def test_exact_king_context_expansion_preserves_evaluations(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            original = pathlib.Path(directory) / "shared.nnue"
            torch.manual_seed(61007)
            shared = Net(16, buckets=8, king_buckets=1)
            with torch.no_grad():
                shared.ft_bias.fill_(.35)
                shared.out.bias.fill_(.25)
            shared.export(original)
            expanded = pathlib.Path(directory) / "expanded.nnue"
            net = Net(16, buckets=8, king_buckets=64)
            net.initialize_quantized(original)
            net.export(expanded)
            fens = []
            # Both turns, with every perspective king square represented.
            for king in range(64):
                board = {king: "K", (king + 32) % 64: "k"}
                rows = []
                for rank in range(7, -1, -1):
                    row, empty = "", 0
                    for file in range(8):
                        piece = board.get(rank * 8 + file)
                        if piece:
                            row += (str(empty) if empty else "") + piece
                            empty = 0
                        else:
                            empty += 1
                    rows.append(row + (str(empty) if empty else ""))
                fens.extend("/".join(rows) + f" {side} - - 0 1" for side in ("w", "b"))
            for fen in fens:
                raw = fen_to_record(fen).view(np.uint8).reshape(1, 32).copy()
                shared = decode(torch.from_numpy(raw), 1)
                contextual = decode(torch.from_numpy(raw), 64)
                for before, after in zip(shared[:2], contextual[:2]):
                    self.assertTrue(torch.equal(before, after % 768))
                values = []
                for path in (original, expanded):
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        check(path, fen)
                    values.append(output.getvalue())
                self.assertEqual(*values)


if __name__ == "__main__":
    unittest.main()
