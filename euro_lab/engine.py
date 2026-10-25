from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import skellam, poisson
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = pd.Timestamp("2024-06-14")
ALIASES = {
    "Czechia": "Czech Republic",
    "Türkiye": "Turkey",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "USA": "United States",
}
HOME_FEATURES = [
    "home_weight_goals",
    "away_weight_goals_suf",
    "home_weight_points",
    "away_weight_points",
]
AWAY_FEATURES = [
    "away_weight_goals",
    "home_weight_goals_suf",
    "home_weight_points",
    "away_weight_points",
]
GROUPS = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Slovenia", "Denmark", "Serbia", "England"],
    "D": ["Poland", "Netherlands", "Austria", "France"],
    "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "F": ["Turkey", "Georgia", "Portugal", "Czech Republic"],
}
TEAMS = sorted(sum(GROUPS.values(), []))
# Columns are opponents of group winners B, C, E, F (UEFA 2022–24, 21.05).
THIRDS = dict(
    zip(
        [
            "ABCD",
            "ABCE",
            "ABCF",
            "ABDE",
            "ABDF",
            "ABEF",
            "ACDE",
            "ACDF",
            "ACEF",
            "ADEF",
            "BCDE",
            "BCDF",
            "BCEF",
            "BDEF",
            "CDEF",
        ],
        [
            "ADBC",
            "AEBC",
            "AFBC",
            "DEAB",
            "DFAB",
            "EFBA",
            "EDCA",
            "FDCA",
            "EFCA",
            "EFDA",
            "EDBC",
            "FDCB",
            "FECB",
            "FEDB",
            "FEDC",
        ],
    )
)


def load_data():
    matches = pd.read_csv(ROOT / "data/results.csv.gz", parse_dates=["date"])
    ranks = pd.read_csv(ROOT / "data/rankings.csv.gz", parse_dates=["rank_date"])
    for col in ["home_team", "away_team"]:
        matches[col] = matches[col].replace(ALIASES)
    ranks["team"] = ranks.country_full.replace(ALIASES)
    return matches, ranks


def attach_rankings(matches, rankings):
    """Backward-only as-of joins retain actual ranking release dates."""
    out = matches.sort_values("date").copy()
    for side in ["home", "away"]:
        rs = (
            rankings[["rank_date", "team", "rank", "total_points"]]
            .drop_duplicates(["team", "rank_date"])
            .rename(
                columns={
                    "team": f"{side}_team",
                    "rank": f"rank_{side}",
                    "total_points": f"points_{side}",
                    "rank_date": f"rank_date_{side}",
                }
            )
        )
        out = pd.merge_asof(
            out.sort_values("date"),
            rs.sort_values(f"rank_date_{side}"),
            left_on="date",
            right_on=f"rank_date_{side}",
            by=f"{side}_team",
            direction="backward",
        )
    return out


def build_features(matches, rankings):
    """Build match features before adding any result from that calendar date."""
    pre = matches[matches.date < CUTOFF].copy()
    ranked = attach_rankings(pre, rankings)
    history, rows = {}, []
    for date, daily in ranked.groupby("date", sort=True):
        for row in daily.to_dict("records"):
            for side in ["home", "away"]:
                hist = history.get(row[f"{side}_team"], [])
                row[f"{side}_games"] = len(hist)
                row[f"{side}_history_end"] = hist[-1][3] if hist else pd.NaT
                for field, idx in [("goals", 0), ("conceded", 1), ("form_points", 2)]:
                    row[f"{side}_{field}"] = (
                        np.mean([g[idx] for g in hist]) if hist else np.nan
                    )
                    row[f"{side}_{field}_l5"] = (
                        np.mean([g[idx] for g in hist[-5:]]) if hist else np.nan
                    )
            row["rank_dif"] = row["rank_home"] - row["rank_away"]
            for side, other in [("home", "away"), ("away", "home")]:
                row[f"{side}_weight_goals"] = row[f"{side}_goals"] * row["rank_dif"]
                row[f"{side}_weight_goals_suf"] = (
                    row[f"{side}_conceded"] * row["rank_dif"]
                )
                row[f"{side}_weight_points"] = (
                    row[f"{side}_form_points"] * row["rank_dif"]
                )
            rows.append(row)
        for row in daily.to_dict("records"):
            h, a = row["home_score"], row["away_score"]
            for side, gf, ga in [("home", h, a), ("away", a, h)]:
                history.setdefault(row[f"{side}_team"], []).append(
                    (gf, ga, 3 if gf > ga else 1 if gf == ga else 0, date)
                )
    features = pd.DataFrame(rows)
    eligible = features[
        (features.date >= "2019-08-01")
        & (features.home_games >= 6)
        & (features.away_games >= 6)
    ].dropna(subset=HOME_FEATURES + AWAY_FEATURES)
    return features, eligible.reset_index(drop=True), history


def estimator(name, trees=120, depth=8):
    if name == "Random forest":
        return RandomForestRegressor(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=1,
        )
    if name == "Gradient boosting":
        return GradientBoostingRegressor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
            loss="squared_error",
            random_state=42,
        )
    if name == "Linear regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    return DummyRegressor(strategy="mean")


def fit_pair(frame, name, trees=120, depth=8):
    models = []
    for side, cols in [("home", HOME_FEATURES), ("away", AWAY_FEATURES)]:
        model = estimator(name, trees, depth)
        model.fit(frame[cols].to_numpy(), frame[f"{side}_score"].to_numpy())
        models.append(model)
    return tuple(models)


def predict_pair(models, frame):
    # A goal intensity must be positive; retain the same guard for every model.
    return np.column_stack(
        [
            np.clip(m.predict(frame[c].to_numpy()), 0.05, 6)
            for m, c in zip(models, [HOME_FEATURES, AWAY_FEATURES])
        ]
    )


def outcome_probs(home_rate, away_rate):
    return np.stack(
        [
            skellam.sf(0, home_rate, away_rate),
            skellam.pmf(0, home_rate, away_rate),
            skellam.cdf(-1, home_rate, away_rate),
        ],
        axis=-1,
    )


def score_metrics(actual, predicted):
    probs = outcome_probs(predicted[:, 0], predicted[:, 1])
    outcomes = np.where(
        actual[:, 0] > actual[:, 1], 0, np.where(actual[:, 0] == actual[:, 1], 1, 2)
    )
    return {
        "Goal MAE": float(mean_absolute_error(actual, predicted)),
        "Goal RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "1X2 Brier": float(np.mean(np.sum((probs - np.eye(3)[outcomes]) ** 2, axis=1))),
        "1X2 log loss": float(
            -np.mean(
                np.log(np.maximum(probs[np.arange(len(outcomes)), outcomes], 1e-12))
            )
        ),
    }


def benchmark(frame, trees=120, depth=8):
    train = frame[frame.date < "2023-01-01"]
    valid = frame[(frame.date >= "2023-01-01") & (frame.date < "2024-01-01")]
    rows = []
    for name in [
        "Mean baseline",
        "Linear regression",
        "Random forest",
        "Gradient boosting",
    ]:
        pred = predict_pair(fit_pair(train, name, trees, depth), valid)
        rows.append(
            {
                "Model": name,
                **score_metrics(valid[["home_score", "away_score"]].to_numpy(), pred),
            }
        )
    return pd.DataFrame(rows).sort_values("Goal RMSE").reset_index(drop=True)


def evaluate(frame, name, trees=120, depth=8):
    train, test = (
        frame[frame.date < "2024-01-01"],
        frame[frame.date >= "2024-01-01"].copy(),
    )
    models = fit_pair(train, name, trees, depth)
    pred = predict_pair(models, test)
    test[["pred_home", "pred_away"]] = pred
    p = outcome_probs(pred[:, 0], pred[:, 1])
    test[["p_home", "p_draw", "p_away"]] = p
    return (
        models,
        test,
        score_metrics(test[["home_score", "away_score"]].to_numpy(), pred),
    )


def team_snapshot(history, rankings):
    latest = (
        rankings[rankings.rank_date < CUTOFF]
        .sort_values("rank_date")
        .drop_duplicates("team", keep="last")
        .set_index("team")
    )
    rows = []
    for team in TEAMS:
        h = history[team]
        rows.append(
            {
                "Team": team,
                "FIFA rank": int(latest.loc[team, "rank"]),
                "Goals scored": np.mean([x[0] for x in h]),
                "Goals conceded": np.mean([x[1] for x in h]),
                "Points per game": np.mean([x[2] for x in h]),
                "Last-five goals": np.mean([x[0] for x in h[-5:]]),
                "Matches": len(h),
                "Latest result": h[-1][3],
            }
        )
    return pd.DataFrame(rows).set_index("Team")


def pair_features(snapshot, home, away):
    h, a = snapshot.loc[home], snapshot.loc[away]
    d = h["FIFA rank"] - a["FIFA rank"]
    return pd.DataFrame(
        [
            {
                "home_weight_goals": h["Goals scored"] * d,
                "away_weight_goals": a["Goals scored"] * d,
                "home_weight_goals_suf": h["Goals conceded"] * d,
                "away_weight_goals_suf": a["Goals conceded"] * d,
                "home_weight_points": h["Points per game"] * d,
                "away_weight_points": a["Points per game"] * d,
            }
        ]
    )


def match_rates(models, snapshot, home, away):
    """Average both administrative orientations for neutral EURO fixtures."""
    forward = predict_pair(models, pair_features(snapshot, home, away))[0]
    reverse = predict_pair(models, pair_features(snapshot, away, home))[0][::-1]
    return (forward + reverse) / 2


def goal_grid(h, a, maximum=8):
    return np.outer(
        poisson.pmf(np.arange(maximum + 1), h), poisson.pmf(np.arange(maximum + 1), a)
    )


def rank_group(teams, scores, rng):
    totals = {t: [0, 0, 0, 0] for t in teams}  # points, goals for, goals against, wins
    for h, a, hg, ag in scores:
        for t, gf, ga in [(h, hg, ag), (a, ag, hg)]:
            totals[t][0] += 3 if gf > ga else 1 if gf == ga else 0
            totals[t][1] += gf
            totals[t][2] += ga
            totals[t][3] += int(gf > ga)

    def resolve(tied):
        if len(tied) < 2:
            return tied
        mini = {t: [0, 0, 0] for t in tied}
        for h, a, hg, ag in scores:
            if h in tied and a in tied:
                for t, gf, ga in [(h, hg, ag), (a, ag, hg)]:
                    mini[t][0] += 3 if gf > ga else 1 if gf == ga else 0
                    mini[t][1] += gf - ga
                    mini[t][2] += gf
        keys = {t: tuple(mini[t]) for t in tied}
        buckets = {k: [t for t in tied if keys[t] == k] for k in set(keys.values())}
        if len(buckets) > 1:
            return sum([resolve(buckets[k]) for k in sorted(buckets, reverse=True)], [])
        # No cards/qualifier ranking model: unresolved ties use a seeded draw.
        lottery = {t: rng.random() for t in tied}
        return sorted(
            tied,
            key=lambda t: (totals[t][1] - totals[t][2], totals[t][1], lottery[t]),
            reverse=True,
        )

    order = []
    for points in sorted(set(totals[t][0] for t in teams), reverse=True):
        order += resolve([t for t in teams if totals[t][0] == points])
    return order, totals


def round_of_16(orders, third_groups):
    mapping = dict(zip("BCEF", THIRDS["".join(sorted(third_groups))]))
    w = lambda g: orders[g][0]
    r = lambda g: orders[g][1]
    third = lambda g: orders[mapping[g]][2]
    return [
        (w("B"), third("B")),
        (w("A"), r("C")),
        (w("F"), third("F")),
        (r("D"), r("E")),
        (w("E"), third("E")),
        (w("D"), r("F")),
        (w("C"), third("C")),
        (r("A"), r("B")),
    ]


def simulate(rate_matrix, n=1000, seed=2024):
    rng = np.random.default_rng(seed)
    counts = {t: np.zeros(5, dtype=int) for t in TEAMS}
    group_points = {t: np.zeros(10, dtype=int) for t in TEAMS}
    last_bracket = []

    def knockout(h, a):
        rh, ra = rate_matrix[h, a]
        hg, ag = rng.poisson([rh, ra])
        if hg == ag:
            eh, ea = rng.poisson([rh / 3, ra / 3])
            hg += eh
            ag += ea
        return h if hg > ag else a if ag > hg else (h if rng.random() < 0.5 else a)

    for run in range(n):
        orders, totals = {}, {}
        for group, teams in GROUPS.items():
            scores = [
                (h, a, *rng.poisson(rate_matrix[h, a]))
                for h, a in combinations(teams, 2)
            ]
            orders[group], totals[group] = rank_group(teams, scores, rng)
            for t in teams:
                group_points[t][totals[group][t][0]] += 1
        thirds = [(g, orders[g][2]) for g in GROUPS]
        lottery = {g: rng.random() for g, _ in thirds}
        thirds.sort(
            key=lambda gt: (
                totals[gt[0]][gt[1]][0],
                totals[gt[0]][gt[1]][1] - totals[gt[0]][gt[1]][2],
                totals[gt[0]][gt[1]][1],
                totals[gt[0]][gt[1]][3],
                lottery[gt[0]],
            ),
            reverse=True,
        )
        ties = round_of_16(orders, [g for g, _ in thirds[:4]])
        stages = []
        for stage in range(4):
            for h, a in ties:
                counts[h][stage] += 1
                counts[a][stage] += 1
            winners = [knockout(h, a) for h, a in ties]
            stages.append(
                [
                    {
                        "stage": [
                            "Round of 16",
                            "Quarter-final",
                            "Semi-final",
                            "Final",
                        ][stage],
                        "team_a": h,
                        "team_b": a,
                        "winner": w,
                    }
                    for (h, a), w in zip(ties, winners)
                ]
            )
            if stage == 3:
                counts[winners[0]][4] += 1
            else:
                ties = list(zip(winners[::2], winners[1::2]))
        if run == n - 1:
            last_bracket = sum(stages, [])
    table = (
        pd.DataFrame(
            counts,
            index=["Round of 16", "Quarter-final", "Semi-final", "Final", "Champion"],
        ).T
        / n
    )
    table.index.name = "Team"
    table["MC standard error"] = np.sqrt(table.Champion * (1 - table.Champion) / n)
    return (
        table.sort_values("Champion", ascending=False),
        pd.DataFrame(group_points, index=range(10)).T / n,
        pd.DataFrame(last_bracket),
    )


def all_rates(models, snapshot):
    pairs = [(h, a) for h in TEAMS for a in TEAMS if h != a]
    frame = pd.concat(
        [pair_features(snapshot, h, a) for h, a in pairs], ignore_index=True
    )
    raw = dict(zip(pairs, predict_pair(models, frame)))
    return {(h, a): (raw[h, a] + raw[a, h][::-1]) / 2 for h, a in pairs}
