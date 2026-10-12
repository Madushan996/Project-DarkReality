"""Independent integer arithmetic, gradient and accepted-engine checks."""
import contextlib
import io
import json
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import train
from train_qat import QuantizedNet, rounded, truncated

ROOT = pathlib.Path(__file__).resolve().parent.parent


def predict(net, fens):
    raw = np.concatenate([train.fen_to_record(fen) for fen in fens])
    args = train.decode(torch.from_numpy(raw.view(np.uint8).reshape(-1, 32).copy()), net.king_buckets)
    stm, other, mask, _, _, _, pieces = args
    return net(stm, other, mask, train.output_bucket(pieces, net.buckets))


class QuantizationTests(unittest.TestCase):
    def test_signed_rounding_and_derivatives(self):
        values = torch.tensor([-.5, .5, 1.5, -1.5, -1.9, 1.9], requires_grad=True)
        self.assertEqual(rounded(values).tolist(), [0., 0., 2., -2., -2., 2.])
        self.assertEqual(truncated(values).tolist(), [0., 0., 1., -1., -1., 1.])
        (rounded(values) + truncated(values)).sum().backward()
        self.assertEqual(values.grad.tolist(), [2.] * 6)

    def test_exported_integer_forward_independently(self):
        torch.manual_seed(61007)
        net = QuantizedNet(5, 8, 1)
        with torch.no_grad():
            net.ft_bias.copy_(torch.tensor([.5, -.1, 1.1, .4, .2]))
            net.out.weight.uniform_(-1.8, 1.8)
            net.out.bias.uniform_(-.5, .5)
        fens = [
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1',
            '7k/5Q1p/5K2/8/8/8/8/8 w - - 0 1',
            '8/8/1P6/6pp/R1K3k1/8/5r2/8 b - - 0 51']
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / 'test.nnue'
            net.export(path)
            expected = []
            for fen in fens:
                capture = io.StringIO()
                with contextlib.redirect_stdout(capture):
                    train.check(path, fen)
                expected.append(int(capture.getvalue().strip()))
        actual = torch.round(predict(net, fens).detach() * train.SCALE).to(torch.int64).tolist()
        self.assertEqual(actual, expected)
        loss = torch.sigmoid(predict(net, fens)).square().mean()
        loss.backward()
        for parameter in net.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(net.ft.weight.grad.abs().sum().item(), 0)
        self.assertGreater(net.out.weight.grad.abs().sum().item(), 0)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0)
        before = net.ft.weight.detach().clone()
        opt.step()
        net.clip()
        self.assertFalse(torch.equal(before, net.ft.weight))

    @unittest.skipUnless(os.environ.get('QAT_ACCEPTED_ENGINE'), 'requires exact accepted engine')
    def test_actual_accepted_engine_all_64_fens(self):
        fens = [json.loads(line)['fen'] for line in
                (ROOT / 'data/lichess-pilot.drp.samples.jsonl').read_text().splitlines()]
        net = QuantizedNet(256, 8, 64)
        net.initialize_quantized(ROOT / 'nets/current.nnue')
        command = 'uci\n' + ''.join(f'position fen {fen}\neval\n' for fen in fens) + 'quit\n'
        result = subprocess.run([os.environ['QAT_ACCEPTED_ENGINE']], input=command,
                                capture_output=True, text=True, check=True, timeout=90,
                                creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
        expected = list(map(int, re.findall(r'network units (-?\d+)', result.stdout)))
        self.assertEqual(len(expected), 64)
        with torch.no_grad():
            actual = torch.round(predict(net, fens) * train.SCALE).to(torch.int64).tolist()
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
