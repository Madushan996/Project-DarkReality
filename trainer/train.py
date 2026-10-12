"""DarkReality NNUE trainer.

Trains (768 -> HIDDEN)x2 -> 1 with SCReLU on `.drp` position files written by
`darkreality expand`, and exports the quantised `.nnue` format the engine
embeds (see engine/src/nnue/mod.rs for the byte layout).

    python train.py --data gen1.drp --out runs/gen1 --hidden 128 --epochs 30
    python train.py --check net.nnue --fen "<fen>"     # integer eval, must match engine `eval`
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn

INPUTS = 768
QA, QB, SCALE = 255, 64, 400
MAX_WEIGHT = 1.98  # keeps |quantised output weight| <= 127, so engine i32 sums cannot overflow
MAGIC_V1, MAGIC_V2, MAGIC_V3 = b"DRNNUE01", b"DRNNUE02", b"DRNNUE03"

RECORD = np.dtype(
    [
        ("occ", "<u8"),
        ("pcs", "u1", 16),
        ("score", "<i2"),
        ("result", "u1"),
        ("stm", "u1"),
        ("fullmove", "<u2"),
        ("reserved", "u1", 2),
    ]
)
assert RECORD.itemsize == 32


# --------------------------------------------------------------------- data


def load_raw(paths):
    """All records as one (N, 32) uint8 tensor."""
    raw = np.concatenate([np.fromfile(p, dtype=np.uint8) for p in paths])
    if raw.size % 32:
        raise SystemExit("position file size is not a multiple of 32 bytes")
    return torch.from_numpy(raw.reshape(-1, 32))


def king_bucket_table(device):
    """King bucket by mirrored perspective square (matches the engine)."""
    table = torch.full((64,), 3, dtype=torch.long)
    for sq in range(64):
        rank, file = divmod(sq, 8)
        if rank == 0:
            table[sq] = 0 if file in (0, 1, 6, 7) else 1
        elif rank == 1:
            table[sq] = 2
    return table.to(device)


def decode(raw, king_buckets=1):
    """Decode a (B, 32) uint8 batch into features and labels on its device.

    Four king buckets use horizontal mirroring. With 64 buckets, every
    perspective king square has its own transform and files remain intact.

    Each board yields 64 slots (one per square); empty squares get weight 0,
    which lets EmbeddingBag sum exactly the active features.
    """
    b, device = raw.shape[0], raw.device
    r = raw.long()
    occ = torch.zeros(b, dtype=torch.long, device=device)
    for i in range(8):
        occ |= r[:, i] << (8 * i)  # wraps into the sign bit for h8; we only test bits
    pcs = r[:, 8:24]
    nibbles = torch.stack([pcs & 15, pcs >> 4], dim=2).reshape(b, 32)

    squares = torch.arange(64, device=device)
    bits = (occ.unsqueeze(1) >> squares) & 1
    slot = (torch.cumsum(bits, dim=1) - 1).clamp(min=0)
    piece = torch.gather(nibbles, 1, slot)
    piece_type, color = piece >> 1, piece & 1

    if king_buckets == 1:
        white = color * 384 + piece_type * 64 + squares
        black = (1 - color) * 384 + piece_type * 64 + (squares ^ 56)
    else:
        is_king = (piece_type == 5) & bits.bool()
        wk = (is_king & (color == 0)).long().argmax(dim=1, keepdim=True)
        bk = (is_king & (color == 1)).long().argmax(dim=1, keepdim=True) ^ 56
        if king_buckets == 64:
            w_mirror, b_mirror = torch.zeros_like(wk), torch.zeros_like(bk)
            w_base, b_base = wk * INPUTS, bk * INPUTS
        else:
            table = king_bucket_table(device)
            w_mirror, b_mirror = ((wk & 7) >= 4).long() * 7, ((bk & 7) >= 4).long() * 7
            w_base, b_base = table[wk ^ w_mirror] * 768, table[bk ^ b_mirror] * 768
        white = w_base + color * 384 + piece_type * 64 + (squares ^ w_mirror)
        black = b_base + (1 - color) * 384 + piece_type * 64 + (squares ^ 56 ^ b_mirror)
    black_to_move = r[:, 27].bool()
    stm_feat = torch.where(black_to_move.unsqueeze(1), black, white)
    nstm_feat = torch.where(black_to_move.unsqueeze(1), white, black)

    score = r[:, 24] | (r[:, 25] << 8)
    score = torch.where(score >= 32768, score - 65536, score).float()
    return stm_feat, nstm_feat, bits.float(), score, r[:, 26].float(), black_to_move, bits.sum(dim=1)


def output_bucket(pieces, buckets):
    """Equal-width piece-count bands over 2..=32 (matches the engine)."""
    width = -(-32 // buckets)
    return torch.clamp((pieces - 2).clamp(min=0) // width, max=buckets - 1)


def targets(white_score, white_result, black_to_move, wdl_weight, score_scale=1.0):
    """Blend of search score and game result, both from the side to move's view.

    `score_scale` converts stored scores to network units: 1.0 for data whose
    scores are already network units (gen1-gen3), ln(10) = 2.303 for data
    written by engines that report Elo-style centipawns.
    """
    white_score = white_score * score_scale
    sign = 1.0 - 2.0 * black_to_move.float()
    stm_result = torch.where(black_to_move, 1.0 - white_result / 2.0, white_result / 2.0)
    return (1.0 - wdl_weight) * torch.sigmoid(white_score * sign / SCALE) + wdl_weight * stm_result


def record_targets(raw, score, result, black, wdl, score_scale):
    """Ignore absent outcomes for imported records, including mixed batches."""
    result_weight = torch.where((raw[:, 30] & 1).bool(), 0.0, wdl)
    return targets(score, result, black, result_weight, score_scale)


# -------------------------------------------------------------------- model


def screlu(x):
    return torch.clamp(x, 0.0, 1.0) ** 2


class Net(nn.Module):
    def __init__(self, hidden, buckets=8, king_buckets=1):
        super().__init__()
        self.hidden = hidden
        self.buckets = buckets
        self.king_buckets = king_buckets
        self.ft = nn.EmbeddingBag(INPUTS * king_buckets, hidden, mode="sum")
        self.ft_bias = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(2 * hidden, buckets)
        nn.init.normal_(self.ft.weight, std=0.05)
        nn.init.uniform_(self.out.weight, -1.0 / math.sqrt(2 * hidden), 1.0 / math.sqrt(2 * hidden))
        nn.init.zeros_(self.out.bias)

    def forward(self, stm_feat, nstm_feat, mask, bucket):
        us = self.ft(stm_feat, per_sample_weights=mask) + self.ft_bias
        them = self.ft(nstm_feat, per_sample_weights=mask) + self.ft_bias
        out = self.out(torch.cat([screlu(us), screlu(them)], dim=1))
        return out.gather(1, bucket.unsqueeze(1)).squeeze(1)

    @torch.no_grad()
    def initialize_quantized(self, path):
        """Initialize from this engine's own prior network (never foreign nets)."""
        with open(path, "rb") as source:
            raw = source.read()
        if raw[:8] == MAGIC_V1:
            inputs, hidden, qa, qb, scale = struct.unpack_from("<5I", raw, 8)
            buckets, king_buckets, offset = 1, 1, 28
        elif raw[:8] == MAGIC_V2:
            inputs, hidden, buckets, qa, qb, scale = struct.unpack_from("<6I", raw, 8)
            king_buckets, offset = 1, 32
        elif raw[:8] == MAGIC_V3:
            inputs, hidden, buckets, king_buckets, qa, qb, scale = struct.unpack_from("<7I", raw, 8)
            offset = 36
        else:
            raise ValueError("not an original DarkReality network")
        expand_context = king_buckets == 1 and self.king_buckets == 64
        if (inputs, hidden, qa, qb, scale) != (INPUTS*king_buckets, self.hidden, QA, QB, SCALE) or (king_buckets != self.king_buckets and not expand_context):
            raise ValueError("initial network feature architecture differs")
        if buckets not in (1, self.buckets):
            raise ValueError("initial network output buckets differ")
        expected_size = offset + inputs*hidden*2 + hidden*2 + buckets*hidden*4 + buckets*4
        if len(raw) != expected_size:
            raise ValueError("invalid network file length")
        def read(dtype, count, shape, divisor):
            nonlocal offset
            value = np.frombuffer(raw, dtype, count, offset).reshape(shape).astype(np.float32) / divisor
            offset += count * np.dtype(dtype).itemsize
            return torch.from_numpy(value).to(self.ft.weight.device)
        ft = read("<i2", inputs*hidden, (inputs, hidden), QA)
        self.ft.weight.copy_(ft.repeat(64, 1) if expand_context else ft)
        self.ft_bias.copy_(read("<i2", hidden, (hidden,), QA))
        self.out.weight.copy_(read("<i2", buckets*2*hidden, (buckets, 2*hidden), QB).expand(self.buckets, -1))
        self.out.bias.copy_(read("<i4", buckets, (buckets,), QA*QB).expand(self.buckets))

    @torch.no_grad()
    def clip(self):
        for p in (self.ft.weight, self.ft_bias, self.out.weight):
            p.clamp_(-MAX_WEIGHT, MAX_WEIGHT)

    @torch.no_grad()
    def export(self, path):
        ft_w = torch.round(self.ft.weight * QA).to(torch.int16).cpu().numpy()  # [768][hidden]
        ft_b = torch.round(self.ft_bias * QA).to(torch.int16).cpu().numpy()
        out_w = torch.round(self.out.weight * QB).to(torch.int16).cpu().numpy()  # [buckets][2*hidden], stm first
        out_b = torch.round(self.out.bias * QA * QB).to(torch.int32).cpu().numpy()
        with open(path, "wb") as f:
            if self.king_buckets == 1:
                f.write(MAGIC_V2)
                f.write(struct.pack("<6I", INPUTS, self.hidden, self.buckets, QA, QB, SCALE))
            else:
                f.write(MAGIC_V3)
                f.write(struct.pack("<7I", INPUTS * self.king_buckets, self.hidden, self.buckets, self.king_buckets, QA, QB, SCALE))
            f.write(ft_w.astype("<i2").tobytes())
            f.write(ft_b.astype("<i2").tobytes())
            f.write(out_w.astype("<i2").tobytes())
            f.write(out_b.astype("<i4").tobytes())


# ----------------------------------------------------------------- training


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.stream:
        from stream_records import RecordStream
        stream = RecordStream(args.data, block_records=args.stream_block, seed=args.seed)
        positions = stream.positions
        val_arrays = list(stream.batches(65536, validation=True, max_positions=min(65536, max(positions // 100, 1))))
        if not val_arrays:
            raise ValueError("stream has no board-grouped validation positions")
        val = torch.from_numpy(np.concatenate(val_arrays)).to(device)
        data = None
        train_count = max(1, int(positions * .99))
        print(f"{positions:,} positions, board-grouped streaming split, {len(val):,} validation samples on {device}")
    else:
        data = load_raw(args.data)
    # Keep the dataset on the GPU when it fits: batches are then pure device
    # gathers with no host round-trip.
    if data is not None and device.type == "cuda" and data.numel() <= args.gpu_data_gb * 2**30:
        data = data.to(device)
    if data is not None:
        positions = len(data)
        perm = torch.randperm(len(data), device=data.device)
        n_val = min(max(len(data) // 100, 1), 1_000_000)
        val, train_idx = data[perm[:n_val]], perm[n_val:]
        train_count = len(train_idx)
        print(f"{len(data):,} positions ({train_count:,} train, {n_val:,} validation) on {device}, data on {data.device}")

    net = Net(args.hidden, args.buckets, args.king_buckets).to(device)
    if args.init:
        net.initialize_quantized(args.init)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.0)
    steps_per_epoch = max(train_count // args.batch, 1)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: args.final_lr / args.lr + (1 - args.final_lr / args.lr) * 0.5 * (1 + math.cos(math.pi * min(s, total_steps) / total_steps))
    )
    first_epoch = 1
    skip_batches = 0
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        expected = {key: getattr(args, key) for key in
                    ("hidden", "buckets", "king_buckets", "epochs", "batch", "seed", "score_scale", "wdl_start", "wdl_end", "lr", "final_lr", "power", "stream", "stream_block")}
        comparison = dict(expected)
        if args.extend:
            if args.epochs <= saved["configuration"]["epochs"]:
                raise ValueError("extension must increase the planned epochs")
            comparison["epochs"] = saved["configuration"]["epochs"]
            if args.wdl_start != args.wdl_end:
                raise ValueError("extension requires constant outcome blending")
        if saved["configuration"] != comparison or saved["positions"] != positions:
            raise ValueError("resume configuration/data count differs from checkpoint")
        net.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["cpu_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
        first_epoch = saved["epoch"] + (not saved.get("partial_epoch", False))
        skip_batches = saved.get("batches_done", 0) if saved.get("partial_epoch", False) else 0

    cached = None
    if args.cache_gpu:
        if data is not None or device.type != "cuda":
            raise ValueError("GPU cache requires CUDA and --stream")
        free_bytes, _ = torch.cuda.mem_get_info()
        if positions * 32 + 2 * 2**30 > free_bytes:
            raise ValueError("GPU cache leaves less than 2 GiB for training")
        from cache_records import CachedRecords
        cached = CachedRecords(stream, device)

    def batch_loss(raw, wdl):
        raw = raw.to(device, non_blocking=True)
        stm_f, nstm_f, mask, score, result, black, pieces = decode(raw, args.king_buckets)
        # Imported evaluation records have no game outcome. Never turn the
        # placeholder result into a draw target, including in mixed datasets.
        target = record_targets(raw, score, result, black, wdl, args.score_scale)
        pred = torch.sigmoid(net(stm_f, nstm_f, mask, output_bucket(pieces, args.buckets)))
        return torch.mean(torch.abs(pred - target) ** args.power)

    def streaming_batches(epoch):
        if cached is not None:
            for tensor in cached.blocks_for_epoch(epoch):
                for first in range(0, len(tensor), args.batch):
                    yield tensor[first:first+args.batch]
            return
        iterator = iter(stream.blocks_for_epoch(epoch))
        # Keep one CPU block ahead. Transfer a whole bounded block once, then
        # take GPU slices; avoid thousands of tiny per-batch allocations/copies.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(next, iterator, None)
            while (block := future.result()) is not None:
                future = pool.submit(next, iterator, None)
                tensor = torch.from_numpy(block)
                if device.type == "cuda":
                    tensor = tensor.pin_memory().to(device, non_blocking=True)
                for first in range(0, len(tensor), args.batch):
                    yield tensor[first:first+args.batch]
                if device.type == "cuda":
                    torch.cuda.synchronize()  # bound outstanding CPU/GPU blocks

    previous_seconds = 0.0
    last_completed = first_epoch - 1
    best_validation = float("inf")
    for epoch in range(first_epoch, args.epochs + 1):
        if args.deadline_unix and time.time() + previous_seconds * 1.15 + 15 >= args.deadline_unix:
            print(f"budget deadline: finishing after epoch {last_completed}", flush=True)
            break
        # WDL weight ramps linearly from wdl_start to wdl_end over training.
        wdl = args.wdl_start + (args.wdl_end - args.wdl_start) * (epoch - 1) / max(args.epochs - 1, 1)
        if data is not None:
            order = train_idx[torch.randperm(len(train_idx), device=train_idx.device)]
            batches = (data[order[step * args.batch : (step + 1) * args.batch]] for step in range(steps_per_epoch))
        else:
            batches = streaming_batches(epoch)
        net.train()
        start, running = time.time(), torch.zeros((), device=device)
        steps = seen = 0
        for batch_number, raw in enumerate(batches):
            if epoch == first_epoch and batch_number < skip_batches:
                continue
            loss = batch_loss(raw, wdl)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            net.clip()
            running += loss.detach()  # no .item(): avoids a GPU sync every step
            steps += 1
            seen += len(raw)
            if args.log_every and steps % args.log_every == 0:
                elapsed = time.time() - start
                print(f"progress epoch {epoch}/{args.epochs}: {seen:,} records, {steps:,} updates, {seen/max(elapsed, 1e-9):,.0f} pos/s, {elapsed:.1f}s", flush=True)

        net.eval()
        with torch.no_grad():
            val_loss = np.mean([batch_loss(val[i : i + 65536], wdl).item() for i in range(0, len(val), 65536)])
        secs = time.time() - start
        previous_seconds = secs
        last_completed = epoch
        print(
            f"epoch {epoch:3d} | train {running.item() / max(steps, 1):.6f} | val {val_loss:.6f} | wdl {wdl:.2f} "
            f"| lr {sched.get_last_lr()[0]:.2e} | {seen / secs:,.0f} pos/s",
            flush=True,
        )
        torch.save(net.state_dict(), os.path.join(args.out, "checkpoint.pt"))
        net.export(os.path.join(args.out, f"epoch{epoch}.nnue"))
        if val_loss < best_validation:
            best_validation = val_loss
            net.export(os.path.join(args.out, "best.nnue"))
        resume_file = os.path.join(args.out, "resume.pt")
        torch.save({"epoch": epoch, "model": net.state_dict(), "optimizer": opt.state_dict(),
                    "scheduler": sched.state_dict(), "positions": positions,
                    "configuration": {key: getattr(args, key) for key in
                        ("hidden", "buckets", "king_buckets", "epochs", "batch", "seed", "score_scale", "wdl_start", "wdl_end", "lr", "final_lr", "power", "stream", "stream_block")},
                    "cpu_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else []},
                   resume_file + ".tmp")
        os.replace(resume_file + ".tmp", resume_file)
        print(f"saved epoch {epoch}", flush=True)

    net.export(os.path.join(args.out, "final.nnue"))
    import json
    with open(os.path.join(args.out, "training-summary.json"), "w") as output:
        json.dump({"positions": positions, "planned_epochs": args.epochs,
                   "last_completed_epoch": last_completed, "best_validation_loss": best_validation}, output, indent=2)
    print(f"wrote {os.path.join(args.out, 'final.nnue')}")


# ------------------------------------------------------------- verification


def fen_to_record(fen):
    """Packs a FEN the same way the engine's `pack_position` does."""
    board, side = fen.split()[:2]
    squares = {}
    for i, row in enumerate(board.split("/")):
        rank, file = 7 - i, 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                pt = "pnbrqk".index(ch.lower())
                squares[rank * 8 + file] = (pt << 1) | (0 if ch.isupper() else 1)
                file += 1
    rec = np.zeros(1, dtype=RECORD)
    rec["occ"] = sum(1 << sq for sq in squares)
    for i, sq in enumerate(sorted(squares)):
        rec["pcs"][0, i // 2] |= squares[sq] << (4 * (i % 2))
    rec["stm"] = 0 if side == "w" else 1
    return rec


def check(net_path, fen):
    """Integer evaluation replicating the engine exactly."""
    with open(net_path, "rb") as source:
        raw = source.read()
    king_buckets = 1
    if raw[:8] == MAGIC_V1:
        inputs, hidden, qa, qb, scale = struct.unpack_from("<5I", raw, 8)
        buckets, o = 1, 28
    elif raw[:8] == MAGIC_V2:
        inputs, hidden, buckets, qa, qb, scale = struct.unpack_from("<6I", raw, 8)
        o = 32
    else:
        assert raw[:8] == MAGIC_V3
        inputs, hidden, buckets, king_buckets, qa, qb, scale = struct.unpack_from("<7I", raw, 8)
        o = 36
    ft_w = np.frombuffer(raw, "<i2", inputs * hidden, o).reshape(inputs, hidden).astype(np.int64)
    o += inputs * hidden * 2
    ft_b = np.frombuffer(raw, "<i2", hidden, o).astype(np.int64)
    o += hidden * 2
    out_w = np.frombuffer(raw, "<i2", buckets * 2 * hidden, o).reshape(buckets, 2, hidden).astype(np.int64)
    o += buckets * 4 * hidden
    out_b = np.frombuffer(raw, "<i4", buckets, o).astype(np.int64)

    rec = fen_to_record(fen)
    stm_f, nstm_f, mask, _, _, _, pieces = decode(torch.from_numpy(rec.view(np.uint8).reshape(1, 32).copy()), king_buckets)
    b = int(output_bucket(pieces, buckets)[0])
    active = mask[0].bool()
    us = ft_b + ft_w[stm_f[0][active].numpy()].sum(axis=0)
    them = ft_b + ft_w[nstm_f[0][active].numpy()].sum(axis=0)
    s = int((np.clip(us, 0, qa) ** 2 * out_w[b][0]).sum() + (np.clip(them, 0, qa) ** 2 * out_w[b][1]).sum())
    # Engine uses truncating i32 division; mirror it.
    q = int(s / qa)
    print(int((q + int(out_b[b])) * scale / (qa * qb)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", help=".drp files")
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--buckets", type=int, default=8, help="output buckets by piece count")
    p.add_argument("--king-buckets", type=int, default=1, choices=[1, 4, 64], help="input contexts: 1 shared, 4 mirrored regions, or 64 exact king squares")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--final-lr", type=float, default=1e-5)
    p.add_argument("--wdl-start", type=float, default=0.9)
    p.add_argument("--wdl-end", type=float, default=0.9)
    p.add_argument("--power", type=float, default=2.5)
    p.add_argument("--score-scale", type=float, default=1.0,
                   help="multiply stored scores into network units (2.303 for centipawn data)")
    p.add_argument("--gpu-data-gb", type=float, default=12.0, help="keep the dataset on the GPU up to this size")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", help="resume an original trainer state bundle")
    p.add_argument("--extend", action="store_true", help="continue an existing checkpoint with more planned epochs and the stretched cosine schedule")
    p.add_argument("--init", help="initialize from this engine's own quantized network; no foreign networks")
    p.add_argument("--stream", action="store_true", help="bounded-memory shard training with board-grouped validation")
    p.add_argument("--cache-gpu", action="store_true", help="cache all raw records on CUDA; still shuffle bounded blocks")
    p.add_argument("--log-every", type=int, default=1000, help="print progress every this many batches; 0 disables")
    p.add_argument("--deadline-unix", type=float, default=0, help="finish at an epoch boundary before this absolute time")
    p.add_argument("--stream-block", type=int, default=524288, help="maximum records read/shuffled at once")
    p.add_argument("--check", help="network to evaluate with --fen")
    p.add_argument("--fen")
    args = p.parse_args()
    if args.check:
        check(args.check, args.fen)
    else:
        train(args)


if __name__ == "__main__":
    main()
