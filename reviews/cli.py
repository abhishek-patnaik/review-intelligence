"""Command line entry point.

    python -m reviews check            confirm the data and the local model are ready
    python -m reviews smoke --n 20     classify a few random reviews and print them
"""

from __future__ import annotations

import argparse
import time

from reviews import data
from reviews.config import load_config


def cmd_check(args) -> None:
    cfg = load_config()
    tables = data.load_raw(cfg)
    print("Data OK: " + ", ".join(f"{k} {len(v):,}" for k, v in tables.items()))
    from reviews.llm import Classifier

    Classifier(cfg).check()
    print(f"Model OK: {cfg['llm']['model']} is installed and Ollama is running")


def cmd_smoke(args) -> None:
    cfg = load_config()
    from reviews.llm import Classifier

    clf = Classifier(cfg)
    clf.check()
    r = data.load_raw(cfg)["reviews"]
    r = r[r.review_comment_message.notna()].sample(args.n, random_state=args.seed)
    start = time.time()
    for _, row in r.iterrows():
        text = " ".join(str(row.review_comment_message).split())
        themes = clf.classify(text)
        print(f"[{row.review_score}*] {text[:110]}")
        print(f"      -> {', '.join(themes) or '(no complaint)'}")
    secs = time.time() - start
    print(f"\n{args.n} reviews in {secs:.1f}s ({secs / args.n:.2f}s each, cached ones are instant)")


def main() -> None:
    p = argparse.ArgumentParser(prog="reviews")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    s = sub.add_parser("smoke")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_smoke)
    args = p.parse_args()
    args.func(args)
