"""Command line entry point.

    python -m reviews check            confirm the data and the local model are ready
    python -m reviews smoke --n 20     classify a few random reviews and print them
    python -m reviews sample           draw the hand labelling sample and translate it
    python -m reviews label            open the labelling tool in the browser
    python -m reviews label --test     label only the 100 hand labelled test reviews
    python -m reviews evaluate         score the model on the dev split while tuning the prompt
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


def cmd_sample(args) -> None:
    from reviews import labels

    cfg = load_config()
    if labels.SAMPLE_FILE.exists() and not args.force:
        raise SystemExit(
            f"{labels.SAMPLE_FILE.name} already exists. Redrawing would throw away the link "
            "to labels you have given. Use --force only if you really mean it.")
    lc = cfg["labels"]
    pool = labels.reviews_with_text(data.load_raw(cfg))
    sample = labels.draw_sample(pool, lc["per_score"], lc["dev_share"], lc["random_seed"])
    print(f"Drew {len(sample)} reviews from {len(pool):,} with text "
          f"({(sample.split == 'dev').sum()} dev, {(sample.split == 'test').sum()} test)")
    from reviews.translate import Translator

    start = time.time()
    sample["text_en"] = Translator(cfg).translate(sample.text.tolist())
    print(f"Translated in {time.time() - start:.0f}s")
    labels.LABEL_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_csv(labels.SAMPLE_FILE, index=False)
    print(f"Wrote {labels.SAMPLE_FILE.relative_to(labels.ROOT)}")
    for row in sample.head(3).itertuples():
        print(f"  PT: {row.text[:90]}\n  EN: {row.text_en[:90]}")


def cmd_label(args) -> None:
    from reviews import label_app, labels

    if not labels.SAMPLE_FILE.exists():
        raise SystemExit("No sample yet. Run: python -m reviews sample")
    label_app.serve(load_config(), only_test=args.test)


def cmd_evaluate(args) -> None:
    import json

    from reviews import evaluate as ev
    from reviews.config import paths_for
    from reviews.llm import Classifier

    cfg = load_config()
    if args.model:
        cfg["llm"]["model"] = args.model
    clf = Classifier(cfg)
    clf.check()
    df = ev.load_truth(args.labels)
    if args.split == "human_test":
        df = df[df.human_test.astype(int) == 1]
    elif args.split != "all":
        df = df[df.split == args.split]
    if args.split in ("test", "human_test") and not args.confirm_test:
        raise SystemExit("The test split is scored once, at the very end. Add --confirm-test if that is now.")
    themes = list(cfg["themes"])
    print(f"Scoring {cfg['llm']['model']} (prompt {cfg['llm']['prompt_version']}) on "
          f"{len(df)} {args.split} reviews against {args.labels} labels")
    start = time.time()
    pred = ev.classify_many(clf, df.text.tolist(), cfg["llm"].get("workers", 1))
    print(f"  done in {time.time() - start:.0f}s")
    truth = [[t for t in x.split(";") if t] for x in df.themes]
    table, summary = ev.score(truth, pred, themes)
    tag = f"{cfg['llm']['model'].replace(':', '_')}_{cfg['llm']['prompt_version']}_{args.split}_{args.labels}"
    paths = paths_for("real")
    table.to_csv(paths.tables / f"eval_{tag}.csv", index=False)
    (paths.tables / f"eval_{tag}.json").write_text(json.dumps(summary, indent=2))
    errors = df.assign(truth=[";".join(t) for t in truth], pred=[";".join(p) for p in pred])
    errors = errors[errors.truth != errors.pred][["review_id", "review_score", "text", "text_en", "truth", "pred"]]
    errors.to_csv(paths.tables / f"errors_{tag}.csv", index=False)
    print(table.round(2).to_string(index=False))
    print("\n" + "\n".join(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in summary.items()))
    print(f"\n{len(errors)} reviews where model and labels differ, saved to outputs/tables/errors_{tag}.csv")


def main() -> None:
    p = argparse.ArgumentParser(prog="reviews")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(func=cmd_check)
    s = sub.add_parser("smoke")
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_smoke)
    s = sub.add_parser("sample")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_sample)
    s = sub.add_parser("label")
    s.add_argument("--test", action="store_true")
    s.set_defaults(func=cmd_label)
    s = sub.add_parser("evaluate")
    s.add_argument("--split", default="dev", choices=["dev", "test", "human_test", "all"])
    s.add_argument("--labels", default="working", choices=["working", "human"])
    s.add_argument("--model", default=None, help="override the model in config.toml")
    s.add_argument("--confirm-test", action="store_true")
    s.set_defaults(func=cmd_evaluate)
    args = p.parse_args()
    args.func(args)
