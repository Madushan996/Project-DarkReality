"""Import CC0 Lichess evaluations as supervised-only DarkReality records.

This code packs positions itself; python-chess supplies board/rules validation.
No other engine's implementation, network or trainer is used. Stream a bounded
prefix of the public JSONL export, rather than downloading the entire database.
"""
import argparse
import collections
import contextlib
import hashlib
import io
import json
import pathlib
import random
import struct
import time
import urllib.request

import chess
import zstandard

SOURCE = "https://database.lichess.org/lichess_db_eval.jsonl.zst"


def recover_data(path):
    """Validate and describe a prefix retained after worker interruption.

    Do not invent the missing compressed-stream statistics or input hash.
    Validate all packed fields, then compare the saved source examples exactly.
    """
    import numpy as np
    destination = pathlib.Path(path)
    size = destination.stat().st_size
    if not size or size % 32:
        raise RuntimeError("Interrupted file must contain complete 32-byte records")
    sha = hashlib.sha256()
    accepted = 0
    dtype = np.dtype([("occ", "<u8"), ("pcs", "u1", 16), ("score", "<i2"),
                      ("result", "u1"), ("stm", "u1"), ("fullmove", "<u2"), ("flags", "u1", 2)])
    with open(destination, "rb") as stream:
        while data := stream.read(65536 * 32):
            sha.update(data)
            rows = np.frombuffer(data, dtype=dtype)
            raw = np.frombuffer(data, dtype=np.uint8).reshape(-1, 32)
            counts = np.unpackbits(raw[:, :8], axis=1).sum(axis=1)
            codes = np.stack((rows["pcs"] & 15, rows["pcs"] >> 4), axis=2).reshape(-1, 32)
            used = np.arange(32)[None, :] < counts[:, None]
            valid = ((counts >= 6) & (counts <= 32) & (rows["stm"] <= 1)
                     & (rows["result"] == 1) & (np.abs(rows["score"].astype(np.int32)) <= 1500)
                     & (rows["flags"][:, 0] == 1) & (rows["flags"][:, 1] == 0)
                     & ((codes <= 11) | ~used).all(axis=1)
                     & ((codes == 10) & used).sum(axis=1).__eq__(1)
                     & ((codes == 11) & used).sum(axis=1).__eq__(1))
            if not valid.all():
                raise RuntimeError(f"Invalid recovered record near index {accepted + int(np.flatnonzero(~valid)[0])}")
            accepted += len(rows)
    samples = pathlib.Path(str(destination) + ".samples.jsonl")
    with open(destination, "rb") as records:
        for line in samples.read_text().splitlines():
            packed, _ = convert(json.loads(line))
            if records.read(32) != packed:
                raise RuntimeError("Recovered data differs from saved source examples")
    manifest = {"source": SOURCE, "source_documentation": "https://database.lichess.org/#evals",
                "license": "CC0-1.0", "labels": "Stockfish evaluations distributed by Lichess, white POV centipawns",
                "result_semantics": "unknown; reserved[0] bit 0 means supervised score only",
                "stats": {"accepted": accepted}, "records_sha256": sha.hexdigest(), "created_unix": time.time(),
                "recovery": "validated complete prefix saved by the preempted original importer; stream-end statistics unavailable",
                "filters": {"min_depth": 20, "max_cp": 1500, "min_pieces": 6,
                            "sample_probability": 0.5, "seed": 61007}}
    pathlib.Path(str(destination) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def convert(entry, min_depth=20, max_cp=1500, min_pieces=6):
    """Return (32-byte record, FEN) or an explicit filtering reason.

    Lichess cp labels are white POV. Their PV uses UCI_Chess960 castling.
    Result=1 is a placeholder, not an observed draw: reserved[0] bit 0
    tells our trainer to ignore this field and use only the score target.
    """
    evaluations = [e for e in entry.get("evals", []) if e.get("pvs")]
    if not evaluations:
        return "no_eval"
    best = max(evaluations, key=lambda e: (e.get("depth", 0), e.get("knodes", 0)))
    pv = best["pvs"][0]
    cp = pv.get("cp")
    if best.get("depth", 0) < min_depth:
        return "shallow"
    if type(cp) is not int or abs(cp) > max_cp:
        return "mate_or_extreme"
    try:
        board = chess.Board(entry["fen"])
        if not board.is_valid():
            return "invalid_board"
        if board.is_check() or board.is_game_over():
            return "check_or_terminal"
        if board.occupied.bit_count() < min_pieces:
            return "few_pieces"
        board.chess960 = True
        move = board.parse_uci(pv["line"].split()[0])
        if board.is_capture(move) or move.promotion:
            return "tactical_pv"
    except (ValueError, KeyError, IndexError):
        return "invalid_fen_or_pv"
    occupancy = board.occupied
    packed = bytearray(16)
    squares = chess.scan_forward(occupancy)
    for i, sq in enumerate(squares):
        piece = board.piece_at(sq)
        code = (piece.piece_type - 1) * 2 + (not piece.color)
        packed[i // 2] |= code << (4 * (i % 2))
    record = struct.pack("<Q16shBBHBB", occupancy, bytes(packed), cp,
                         1, not board.turn, min(board.fullmove_number, 65535), 1, 0)
    assert len(record) == 32
    return record, entry["fen"]


class CountedStream(io.RawIOBase):
    def __init__(self, stream):
        self.stream, self.bytes_read = stream, 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        self.bytes_read += len(data)
        buffer[:len(data)] = data
        return len(data)


def import_data(args):
    destination = pathlib.Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    stats = collections.Counter()
    seen = set()
    rng = random.Random(args.seed)
    output_hash, input_hash = hashlib.sha256(), hashlib.sha256()
    start = time.monotonic()
    source = args.source
    with contextlib.ExitStack() as stack:
        if source.startswith("https://"):
            request = urllib.request.Request(source, headers={"User-Agent": "NARC-Reality-NNUE-research/1.0"})
            response = stack.enter_context(urllib.request.urlopen(request, timeout=60))
            source_meta = {key: response.headers.get(key) for key in ("Last-Modified", "ETag", "Content-Length")}
            stream = response
        else:
            stream = stack.enter_context(open(source, "rb"))
            source_meta = {"path": str(pathlib.Path(source).resolve())}
        counted = CountedStream(stream)
        binary = io.BufferedReader(counted)
        if source.endswith(".zst"):
            binary = stack.enter_context(zstandard.ZstdDecompressor().stream_reader(binary))
        lines = stack.enter_context(io.TextIOWrapper(binary, encoding="utf-8"))
        output = stack.enter_context(open(destination, "xb"))
        examples = stack.enter_context(open(str(destination) + ".samples.jsonl", "x"))
        for line in lines:
            if time.monotonic() - start >= args.max_seconds or counted.bytes_read >= args.max_download_gb * 2**30:
                stats["limit_reached"] = 1
                break
            stats["lines_seen"] += 1
            input_hash.update(line.encode())
            if rng.random() > args.sample_probability:
                stats["random_skip"] += 1
                continue
            try:
                entry = json.loads(line)
                converted = convert(entry, args.min_depth, args.max_cp, args.min_pieces)
            except (ValueError, TypeError):
                stats["invalid_json"] += 1
                continue
            if isinstance(converted, str):
                stats[converted] += 1
                continue
            record, fen = converted
            # Include board and side, exclude label and missing FEN counters.
            key = hashlib.blake2b(record[:24] + record[27:28], digest_size=16).digest()
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            output.write(record)
            output_hash.update(record)
            stats["accepted"] += 1
            if stats["accepted"] <= 64:
                examples.write(json.dumps(entry) + "\n")
            if stats["accepted"] % 100_000 == 0:
                print(f"accepted={stats['accepted']:,} lines={stats['lines_seen']:,} downloaded={counted.bytes_read / 2**20:.1f}MiB", flush=True)
            if stats["accepted"] >= args.positions:
                break
        compressed_bytes = counted.bytes_read
    manifest = {"source": source, "source_documentation": "https://database.lichess.org/#evals",
                "license": "CC0-1.0", "labels": "Stockfish evaluations distributed by Lichess, white POV centipawns",
                "result_semantics": "unknown; reserved[0] bit 0 means supervised score only",
                "source_metadata": source_meta, "filters": vars(args), "stats": dict(stats),
                "compressed_bytes_read": compressed_bytes, "seconds": time.monotonic() - start,
                "consumed_jsonl_sha256": input_hash.hexdigest(), "records_sha256": output_hash.hexdigest(),
                "created_unix": time.time(), "original_code": "DarkReality/trainer/import_lichess.py"}
    pathlib.Path(str(destination) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    if not stats["accepted"]:
        raise RuntimeError("No usable training records; do not train this file")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--out", required=True)
    p.add_argument("--recover", action="store_true", help="validate an interrupted prefix instead of downloading")
    p.add_argument("--positions", type=int, default=5_000_000)
    p.add_argument("--min-depth", type=int, default=20)
    p.add_argument("--max-cp", type=int, default=1500)
    p.add_argument("--min-pieces", type=int, default=6)
    p.add_argument("--sample-probability", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=61007)
    p.add_argument("--max-download-gb", type=float, default=8)
    p.add_argument("--max-seconds", type=int, default=3600)
    a = p.parse_args()
    if a.positions < 1 or not 0 < a.sample_probability <= 1 or a.max_seconds < 1 or a.max_download_gb <= 0:
        p.error("position/time/byte limits must be positive; sampling probability must be in (0,1]")
    if a.recover:
        recover_data(a.out)
    else:
        import_data(a)


if __name__ == "__main__":
    main()
