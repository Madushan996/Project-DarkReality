"""Original DR03 quantization-aware continuation of our accepted evaluator.

The forward pass uses exported integer weights and the engine's two
truncating divisions. Rounding has an identity backward derivative so the
existing floating parameters can learn across quantization boundaries.
No architecture change, foreign weights or engine changes are required.
"""
import torch
import torch.nn.functional as F
import train as original


class IntegerStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, rounding):
        return torch.round(value) if rounding else torch.trunc(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def rounded(value):
    return IntegerStep.apply(value, True)


def truncated(value):
    return IntegerStep.apply(value, False)


class QuantizedNet(original.Net):
    def forward(self, stm_feat, nstm_feat, mask, bucket):
        qa, qb = original.QA, original.QB
        weights = rounded(self.ft.weight * qa)
        bias = rounded(self.ft_bias * qa)
        us = F.embedding_bag(stm_feat, weights, mode='sum', per_sample_weights=mask) + bias
        them = F.embedding_bag(nstm_feat, weights, mode='sum', per_sample_weights=mask) + bias
        features = torch.cat([us.clamp(0, qa).square(), them.clamp(0, qa).square()], dim=1)
        # Integer products fit float32 individually, but their signed sum
        # need not. Float64 keeps exact integers through the final divisions.
        out_weights = rounded(self.out.weight * qb)[bucket].to(torch.float64)
        out_bias = rounded(self.out.bias * qa * qb)[bucket].to(torch.float64)
        total = (features.to(torch.float64) * out_weights).sum(dim=1)
        units = truncated((truncated(total / qa) + out_bias) * original.SCALE / (qa * qb))
        return (units / original.SCALE).to(self.ft.weight.dtype)


if __name__ == '__main__':
    original.Net = QuantizedNet
    original.main()
