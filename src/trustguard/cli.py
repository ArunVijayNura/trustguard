"""Command-line entry point.

    trustguard train      # train + save model to artifacts/
    trustguard evaluate   # print comparison table and write reports/evaluation.md
    trustguard score FILE # score a JSON file of listings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .evaluate import compare_systems, recall_by_type
from .model import DEFAULT_MODEL_PATH, RiskModel, train_default_model
from .rules import evaluate_rules, rule_risk
from .scoring import decide, score_frame


def _train(args) -> None:
    model, *_ = train_default_model(n=args.n, seed=args.seed)
    path = model.save(args.model)
    print(f"Saved model v{model.version} -> {path}")


def _evaluate(args) -> None:
    model, test_df, X_test, y_test = train_default_model(n=args.n, seed=args.seed)
    model.save(args.model)
    m_scores = model.predict_proba(X_test)
    r_scores = [rule_risk(evaluate_rules(l)) for l in test_df.to_dict("records")]
    table = compare_systems(y_test, m_scores, pd.Series(r_scores).to_numpy())
    by_type = recall_by_type(score_frame(test_df, model))

    print(table.round(3).to_string())
    print("\nRecall by violation type (hybrid, review threshold):")
    print(by_type.round(3).to_string())

    out = Path("reports")
    out.mkdir(exist_ok=True)
    md = ["# Offline evaluation", "",
          f"Held-out synthetic listings: **{len(test_df):,}** "
          f"({int(y_test.sum())} violations, {y_test.mean():.1%} base rate).", "",
          table.round(3).to_markdown(), "", "## Recall by violation type", "",
          by_type.round(3).to_frame("recall").to_markdown(), ""]
    (out / "evaluation.md").write_text("\n".join(md))
    print(f"\nWrote {out / 'evaluation.md'}")


def _score(args) -> None:
    listings = json.loads(Path(args.file).read_text())
    model = RiskModel.load(args.model) if Path(args.model).exists() else None
    for listing in listings if isinstance(listings, list) else [listings]:
        print(json.dumps(decide(listing, model).to_dict(), indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="trustguard", description="App marketplace policy-risk detection")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in [("train", _train), ("evaluate", _evaluate)]:
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=8000)
        sp.add_argument("--seed", type=int, default=7)
        sp.set_defaults(func=fn)
    sp = sub.add_parser("score")
    sp.add_argument("file")
    sp.set_defaults(func=_score)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
