"""Bounded-memory import of CC0 Gigafish Parquet shards through our Rust rules.

No pretrained network or foreign engine implementation is used. Labels are
depth-10 White POV; unlike the deeper Lichess importer, no quiet PV is supplied.
Full-FEN source deduplication does not imply distinct NNUE boards. Records are
not deduplicated here: billion-entry Python sets would exhaust worker memory.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import urllib.request

SOURCE = "https://huggingface.co/datasets/lukesalamone/gigafish-3.8b-d10"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def import_shard(args):
    import pyarrow.parquet as pq
    out = pathlib.Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = pathlib.Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    name = f"data-{args.shard:05d}.parquet"
    parquet = cache / name
    download_start = time.monotonic()
    url = f"{SOURCE}/resolve/main/{name}"
    if not parquet.exists():
        temporary = cache / (name + ".part")
        with urllib.request.urlopen(url, timeout=120) as source, open(temporary, "wb") as target:
            while block := source.read(1 << 20):
                target.write(block)
        temporary.replace(parquet)
    download_seconds = time.monotonic() - download_start
    input_hash = sha256(parquet)
    start = time.monotonic()
    seen = submitted = 0
    with open(out, "xb") as target:
        p = subprocess.Popen([str(pathlib.Path(args.converter).resolve())], stdin=subprocess.PIPE,
                             stdout=target, stderr=subprocess.PIPE)
        try:
            for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32768, columns=["fen", "eval"]):
                rows = []
                for fen, label in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                    seen += 1
                    if label is None or label.startswith("#"):
                        continue
                    try:
                        cp = round(float(label) * 100)
                    except (ValueError, OverflowError):
                        continue
                    if abs(cp) > 1500:
                        continue
                    rows.append(f"{fen}\t{cp}\n")
                submitted += len(rows)
                p.stdin.write("".join(rows).encode("utf-8"))
            p.stdin.close()
            diagnostics = p.stderr.read().decode("utf-8")
            code = p.wait()
            if code:
                raise RuntimeError(f"converter exited {code}: {diagnostics}")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
    seconds = time.monotonic() - start
    stats = json.loads(diagnostics)
    assert out.stat().st_size == stats["accepted"] * 32
    assert stats["seen"] == submitted
    manifest = {"source": SOURCE, "source_file": name, "source_url": url, "license": "CC0-1.0",
                "labels": "Stockfish 16 depth 10, white POV centipawns; no observed game result",
                "filters": "abs(cp)<=1500, 6..32 pieces, valid engine FEN, <=8 pawns/16 pieces per side, not in check, legal move exists; no quiet-PV filter, no board deduplication",
                "source_rows": seen, "submitted": submitted, "stats": stats,
                "source_bytes": parquet.stat().st_size, "download_seconds": download_seconds,
                "conversion_seconds": seconds, "accepted_per_second": stats["accepted"] / max(seconds, 1e-9),
                "source_sha256": input_hash, "records_sha256": sha256(out),
                "converter_sha256": sha256(args.converter), "created_unix": time.time()}
    pathlib.Path(str(out) + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--cache", default="data/gigafish-cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--converter", default="target/release/position-importer.exe")
    args = parser.parse_args()
    if not 0 <= args.shard < 3920:
        parser.error("shard must be between 0 and 3919")
    import_shard(args)
