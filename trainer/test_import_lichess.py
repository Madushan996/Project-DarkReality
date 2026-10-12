import unittest

import numpy as np
import torch

from trainer.import_lichess import convert
from trainer.train import RECORD, decode, fen_to_record, record_targets


class ImportedLabels(unittest.TestCase):
    def entry(self, side="w"):
        return {"fen": f"4k3/ppp5/8/8/8/8/PPP5/3QK3 {side} - -",
                "evals": [{"depth": 25, "pvs": [{"cp": 700, "line": "d1d2" if side == "w" else "e8f8"}]}]}

    def test_packing_agrees_with_existing_trainer(self):
        for side in ("w", "b"):
            entry = self.entry(side)
            packed, _ = convert(entry)
            expected = fen_to_record(entry["fen"])
            expected["score"] = 700
            expected["result"] = 1
            expected["fullmove"] = 1
            expected["reserved"][0, 0] = 1
            self.assertEqual(packed, expected.tobytes())

    def test_unknown_results_never_become_draw_labels(self):
        records = b"".join(convert(self.entry(side))[0] for side in ("w", "b"))
        raw = torch.from_numpy(np.frombuffer(records, dtype=np.uint8).copy().reshape(2, 32))
        _, _, _, score, result, black, _ = decode(raw)
        actual = record_targets(raw, score, result, black, 0.9, 2.303)
        expected = torch.sigmoid(torch.tensor([700.0, -700.0]) * 2.303 / 400)
        torch.testing.assert_close(actual, expected)
        # A genuine self-play outcome still gets the chosen WDL blend.
        raw[0, 30] = 0
        mixed = record_targets(raw, score, result, black, 0.9, 2.303)
        torch.testing.assert_close(mixed[0], expected[0] * 0.1 + 0.45)
        torch.testing.assert_close(mixed[1], expected[1])

    def test_deepest_pv_selected_and_invalid_labels_filtered(self):
        entry = self.entry()
        entry["evals"].append({"depth": 30, "pvs": [{"cp": 900, "line": "d1d2"}]})
        packed, _ = convert(entry)
        self.assertEqual(int(np.frombuffer(packed, dtype=RECORD)["score"][0]), 900)
        entry["evals"][-1]["pvs"][0] = {"mate": 4, "line": "d1d2"}
        self.assertEqual(convert(entry), "mate_or_extreme")
        entry = self.entry()
        entry["evals"][0]["depth"] = 10
        self.assertEqual(convert(entry), "shallow")
        entry = self.entry()
        entry["evals"][0]["pvs"][0]["line"] = "d1d9"
        self.assertEqual(convert(entry), "invalid_fen_or_pv")


if __name__ == "__main__":
    unittest.main()
