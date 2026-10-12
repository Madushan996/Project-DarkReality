"""An interrupted epoch checkpoint must reproduce uninterrupted training."""
import argparse
import pathlib
import tempfile
import unittest
from unittest.mock import patch
import torch
import train

ROOT = pathlib.Path(__file__).resolve().parent.parent


class ResumeTest(unittest.TestCase):
    def test_resume_reproduces_export(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=ROOT / "data") as directory:
            full, interrupted = pathlib.Path(directory) / "full", pathlib.Path(directory) / "interrupted"
            def args(out, resume=None):
                return argparse.Namespace(data=[ROOT / "data/lichess-pilot.drp"], out=str(out), seed=71,
                    stream=True, stream_block=1000, cache_gpu=False, log_every=0, deadline_unix=0, gpu_data_gb=0, hidden=16, buckets=8, king_buckets=4,
                    lr=.001, final_lr=.00001, epochs=2, batch=512, resume=resume, extend=False, init=None, wdl_start=0.,
                    wdl_end=0., score_scale=2.303, power=2.5)
            with patch("torch.cuda.is_available", return_value=False):
                train.train(args(full))
                replace = train.os.replace
                def stop_after_checkpoint(source, target):
                    replace(source, target)
                    raise InterruptedError("simulated interruption after committed epoch")
                with patch("train.os.replace", side_effect=stop_after_checkpoint):
                    with self.assertRaises(InterruptedError):
                        train.train(args(interrupted))
                train.train(args(interrupted, str(interrupted / "resume.pt")))
            self.assertEqual((full / "final.nnue").read_bytes(), (interrupted / "final.nnue").read_bytes())
            # Extending a saved run must continue its Adam steps and epoch
            # numbering rather than replaying the completed data passes.
            before = torch.load(interrupted / "resume.pt", weights_only=False)
            extended = args(interrupted, str(interrupted / "resume.pt"))
            extended.epochs, extended.extend = 3, True
            with patch("torch.cuda.is_available", return_value=False):
                train.train(extended)
            after = torch.load(interrupted / "resume.pt", weights_only=False)
            self.assertEqual(after["epoch"], 3)
            self.assertEqual(after["configuration"]["epochs"], 3)
            for key, state in after["optimizer"]["state"].items():
                self.assertGreater(state["step"], before["optimizer"]["state"][key]["step"])
                self.assertLess(state["step"], before["optimizer"]["state"][key]["step"] * 2)


if __name__ == "__main__":
    unittest.main()
