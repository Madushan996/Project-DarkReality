"""Cross-check compiled import against saved source FENs and the Python packer."""
import json
import pathlib
import subprocess
import unittest

from import_lichess import convert

ROOT = pathlib.Path(__file__).resolve().parent.parent


class FastImportTest(unittest.TestCase):
    def test_saved_source_records(self):
        lines, expected = [], []
        for line in (ROOT / "data/lichess-pilot.drp.samples.jsonl").read_text().splitlines():
            entry = json.loads(line)
            best = max(entry["evals"], key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
            pv = best["pvs"][0]
            lines.append(f"{entry['fen']}\t{pv['cp']}\t{pv['line'].split()[0]}\n")
            expected.append(convert(entry)[0])
        # Legal castle is quiet; malformed/capture/check positions must be rejected.
        lines.extend(["bad fen\t0\n", "8/8/8/8/8/8/8/8 w - - 0 1\t0\n",
                      "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\t0\te2e5\n"])
        p = subprocess.run([ROOT / "target/release/position-importer.exe"], input="".join(lines).encode(),
                           capture_output=True, check=True)
        self.assertEqual(p.stdout, b"".join(expected))
        self.assertEqual(json.loads(p.stderr)["accepted"], len(expected))


if __name__ == "__main__":
    unittest.main()
