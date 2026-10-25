from html import escape
import json
from .engine import ROOT


def build_report(
    table, test, metrics, euros, euro_metrics, fig, model_name, trees, depth, n, seed
):
    leader = table.index[0]
    prob = table.iloc[0].Champion
    report = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>EURO 2024 model report</title><style>body{{max-width:1050px;margin:48px auto;padding:0 24px;background:#f7f5ef;color:#18263c;font:17px/1.6 system-ui}}h1,h2{{font-family:Georgia,serif;font-weight:400}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #d8d7cf}}small{{color:#536071}}</style><h1>EURO 2024 / Model notes</h1><p>Stephen Ahiabah · retrospective reconstruction</p><h2>{escape(leader)} leads at {prob:.1%}</h2><p>Selected model: {escape(model_name)}. Forest settings: {trees} trees, depth {depth}, minimum leaf 5. Forecast cutoff: 14 June 2024. Simulation: {n:,} runs; seed {seed}.</p><p>This rebuild follows the six original notebooks, using real ranking release dates and chronological validation. It does not preserve an original June 2024 forecast.</p>"""
    report += fig.to_html(full_html=False, include_plotlyjs=True)
    report += "<h2>Stage probabilities</h2>" + table.drop(
        columns=["MC standard error"]
    ).to_html(float_format=lambda x: f"{x:.1%}")
    report += (
        "<h2>Historical holdout</h2><p>"
        + escape(json.dumps(metrics))
        + "</p><p>Fit through December 2023; evaluate January–13 June 2024. "
        + str(len(test))
        + " matches. Brier scores sum over three classes.</p>"
    )
    report += (
        "<h2>Actual EURO group games</h2><p>"
        + escape(json.dumps(euro_metrics))
        + "</p>"
        + euros[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "pred_home",
                "pred_away",
            ]
        ].to_html(index=False, float_format=lambda x: f"{x:.2f}")
    )
    report += (
        """<h2>Assumptions and limitations</h2><p>Independent Poisson goals; no player, injury or tactical features. Some historical goals include extra time. Neutral fixtures average both administrative orderings. Tournament rates stay fixed; unresolved disciplinary/qualifier ties use seeded draws. Extra time uses one-third rates and shootouts are 50/50. Monte Carlo precision is not model certainty.</p><h2>Sources</h2><p><a href="https://github.com/steveaq/ML-2024-Euros-Model">Original notebooks and application</a> · <a href="https://github.com/martj42/international_results">International results</a> · <a href="https://www.kaggle.com/datasets/cashncarry/fifaworldranking">FIFA ranking dataset</a></p><h2>Data provenance</h2><pre>"""
        + escape((ROOT / "data/manifest.json").read_text())
        + "</pre></html>"
    )
    return report
