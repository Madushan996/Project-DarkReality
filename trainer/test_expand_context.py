import pathlib
import tempfile
import unittest
import torch
import numpy as np
from train import Net, decode, fen_to_record, output_bucket
from expand_context import expand_checkpoint

ROOT = pathlib.Path(__file__).resolve().parent.parent


class ExpansionTest(unittest.TestCase):
    def test_complete_checkpoint_preserves_predictions_and_optimizer(self):
        torch.set_num_threads(1)
        torch.manual_seed(61007)
        model = Net(16, 8, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        fens = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "8/8/2k5/8/2K5/8/3Q4/8 b - - 0 1"]
        raw = torch.from_numpy(np.concatenate([fen_to_record(fen).view(np.uint8).reshape(1, 32) for fen in fens]))
        stm, nstm, mask, _, _, _, pieces = decode(raw, 1)
        model(stm, nstm, mask, output_bucket(pieces, 8)).square().mean().backward()
        optimizer.step()
        scheduler.step()
        original = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "configuration": {"hidden": 16, "buckets": 8, "king_buckets": 1},
                    "positions": 1234, "epoch": 7, "cpu_rng": torch.get_rng_state(), "cuda_rng": []}
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            source, target = pathlib.Path(directory) / "source.pt", pathlib.Path(directory) / "target.pt"
            torch.save(original, source)
            expand_checkpoint(source, target)
            expanded = torch.load(target, weights_only=False)
            wider = Net(16, 8, 64)
            wider.load_state_dict(expanded["model"])
            x = decode(raw, 64)
            self.assertTrue(torch.equal(model(stm, nstm, mask, output_bucket(pieces, 8)),
                                        wider(x[0], x[1], x[2], output_bucket(x[6], 8))))
            self.assertEqual(expanded["scheduler"], original["scheduler"])
            self.assertTrue(torch.equal(expanded["cpu_rng"], original["cpu_rng"]))
            self.assertEqual((expanded["epoch"], expanded["positions"]), (7, 1234))
            for parameter_id, state in original["optimizer"]["state"].items():
                after = expanded["optimizer"]["state"][parameter_id]
                self.assertTrue(torch.equal(state["step"], after["step"]))
                for key in ("exp_avg", "exp_avg_sq"):
                    expected = state[key].repeat(64, 1) if state[key].shape == (768, 16) else state[key]
                    self.assertTrue(torch.equal(expected, after[key]))
            original["partial_epoch"] = True
            torch.save(original, source)
            with self.assertRaises(ValueError):
                expand_checkpoint(source, target)


if __name__ == "__main__":
    unittest.main()
