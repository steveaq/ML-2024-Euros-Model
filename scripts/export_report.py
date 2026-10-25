"""Reproduce the presentation report without a running Streamlit session."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import plotly.express as px
from euro_lab.engine import (
    load_data,
    build_features,
    team_snapshot,
    fit_pair,
    all_rates,
    simulate,
    evaluate,
    score_metrics,
)
from euro_lab.report import build_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("euro-2024-report.html"))
    args = parser.parse_args()
    matches, ranks = load_data()
    _, frame, history = build_features(matches, ranks)
    models = fit_pair(frame, "Random forest")
    matrix = all_rates(models, team_snapshot(history, ranks))
    table, _, _ = simulate(matrix, 1000, 2024)
    _, test, metrics = evaluate(frame, "Random forest")
    euros = matches[
        (matches.tournament == "UEFA Euro")
        & (matches.date >= "2024-06-14")
        & (matches.date <= "2024-06-26")
    ].copy()
    pred = np.array([matrix[h, a] for h, a in zip(euros.home_team, euros.away_team)])
    euros[["pred_home", "pred_away"]] = pred
    em = score_metrics(euros[["home_score", "away_score"]].to_numpy(), pred)
    top = table.reset_index().head(10).sort_values("Champion")
    fig = px.bar(
        top,
        x="Champion",
        y="Team",
        orientation="h",
        title="EURO 2024 · reconstructed pre-tournament model",
        color_discrete_sequence=["#2846a6"],
        template="plotly_white",
    )
    fig.update_layout(paper_bgcolor="#f7f5ef", plot_bgcolor="#f7f5ef", height=450)
    fig.update_xaxes(tickformat=".0%")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        build_report(
            table, test, metrics, euros, em, fig, "Random forest", 120, 8, 1000, 2024
        )
    )
    print(args.output)
    print("Holdout:", metrics)
    print("Group review:", em)


if __name__ == "__main__":
    main()
