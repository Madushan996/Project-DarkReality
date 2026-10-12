"""Convert our own match PGNs into DRP records with real game outcomes.

Only games between DarkReality builds are used (any game with a non-DarkReality
player is skipped). Each kept position is the one before a move whose comment
holds our engine's own search score, e.g. `{-1.19/14 0.187s}`: the score is
from the mover's view in pawns. Positions in check, positions where the
played move is a capture or promotion, mate scores and book moves are skipped,
as in the original self-play `expand`.

    python trainer/import_own_games.py --out data/own-games.drp testing/*/games.pgn
"""
import argparse
import hashlib
import json
import re

import chess
import chess.pgn
import numpy as np

from train import RECORD, fen_to_record

COMMENT = re.compile(r"^\s*([+-]?\d+\.\d+)/(\d+)\b")
RESULT = {"1-0": 2, "1/2-1/2": 1, "0-1": 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pgns", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--min-depth", type=int, default=6)
    args = p.parse_args()
    records, seen = [], set()
    stats = dict(games=0, skipped_games=0, positions=0, duplicates=0)
    for path in args.pgns:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                headers = game.headers
                players = (headers.get("White", ""), headers.get("Black", ""))
                result = RESULT.get(headers.get("Result"))
                # Our match runner names engines candidate/opponent; reference
                # matches name the foreign engine explicitly.
                if result is None or any("stormphrax" in name.lower() for name in players):
                    stats["skipped_games"] += 1
                    continue
                stats["games"] += 1
                board = game.board()
                for node in game.mainline():
                    move = node.move
                    match = COMMENT.match(node.comment or "")
                    quiet = not board.is_check() and not board.is_capture(move) and move.promotion is None
                    if match and quiet and int(match.group(2)) >= args.min_depth:
                        pawns = float(match.group(1))
                        if abs(pawns) < 30:  # mate scores are reported separately
                            mover_cp = round(pawns * 100)
                            white_cp = mover_cp if board.turn == chess.WHITE else -mover_cp
                            key = (board.board_fen(), board.turn, result)
                            if key in seen:
                                stats["duplicates"] += 1
                            else:
                                seen.add(key)
                                rec = fen_to_record(board.fen())
                                rec["score"] = max(-32000, min(32000, white_cp))
                                rec["result"] = result
                                rec["fullmove"] = board.fullmove_number
                                records.append(rec)
                    board.push(move)
    data = np.concatenate(records) if records else np.zeros(0, dtype=RECORD)
    assert data.dtype.itemsize == 32
    rng = np.random.default_rng(61007)
    data = data[rng.permutation(len(data))]
    data.tofile(args.out)
    stats["positions"] = int(len(data))
    stats["sha256"] = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    stats["sources"] = len(args.pgns)
    with open(args.out + ".json", "w") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
