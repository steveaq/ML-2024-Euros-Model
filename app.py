"""Run with: streamlit run app.py"""

import json
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import poisson
import streamlit as st
from euro_lab.engine import (
    ROOT,
    TEAMS,
    GROUPS,
    HOME_FEATURES,
    load_data,
    build_features,
    benchmark,
    evaluate,
    fit_pair,
    team_snapshot,
    match_rates,
    goal_grid,
    outcome_probs,
    all_rates,
    simulate,
    score_metrics,
)
from euro_lab.ui import style, header, note, chart, COLORS
from euro_lab.report import build_report

st.set_page_config(page_title="EURO 2024 · Model notes", page_icon="⚽", layout="wide")
style()


@st.cache_data(show_spinner="Preparing the pre-tournament data…")
def prepared():
    matches, ranks = load_data()
    features, eligible, history = build_features(matches, ranks)
    return matches, ranks, features, eligible, team_snapshot(history, ranks)


@st.cache_data(show_spinner="Comparing models on 2023 matches…")
def comparison(trees, depth):
    return benchmark(prepared()[3], trees, depth)


@st.cache_resource(show_spinner="Fitting the selected goal models…")
def fitted(name, trees, depth):
    return fit_pair(prepared()[3], name, trees, depth)


@st.cache_data(show_spinner="Evaluating on the 2024 holdout…")
def validation(name, trees, depth):
    _, test, metrics = evaluate(prepared()[3], name, trees, depth)
    return test, metrics


@st.cache_data(show_spinner="Calculating all neutral-ground matchups…")
def rates(name, trees, depth):
    return all_rates(fitted(name, trees, depth), prepared()[4])


@st.cache_data(show_spinner="Simulating the tournament…")
def tournament(name, trees, depth, n, seed):
    return simulate(rates(name, trees, depth), n, seed)


matches, ranks, features, frame, snapshot = prepared()
with st.sidebar:
    st.html(
        '<div class="eyebrow">Independent research / 2024</div><div class="side-title">EURO<br>Model notes.</div><div class="side-copy">From the source data to the tournament forecast. Built from the original six notebooks.</div>'
    )
    st.caption("Forecast date: 14 June 2024")
    model_name = st.selectbox(
        "Goal model",
        ["Random forest", "Gradient boosting", "Linear regression", "Mean baseline"],
        key="goal_model",
    )
    with st.expander("Random Forest settings"):
        with st.form("training_settings"):
            trees_value = st.select_slider("Trees", options=[80, 120, 200], value=120)
            depth_value = st.select_slider(
                "Maximum depth", options=[4, 6, 8, 12], value=8
            )
            submitted = st.form_submit_button("Apply settings", width="stretch")
        if submitted:
            st.session_state["forest_settings"] = (trees_value, depth_value)
    trees, depth = st.session_state.get("forest_settings", (120, 8))
    st.caption(f"Forest: {trees} trees · depth {depth} · minimum leaf 5")


def overview():
    header(
        "01",
        "What makes a tournament forecast?",
        "The original project estimated each team’s goals, turned those estimates into score probabilities and used them to work through EURO 2024. This app makes each step available to inspect and rerun.",
    )
    cols = st.columns(4)
    for c, label, value in zip(
        cols,
        [
            "Model-ready matches",
            "Teams in the finals",
            "Training cutoff",
            "Original notebooks",
        ],
        [f"{len(frame):,}", "24", "13 Jun ’24", "6"],
    ):
        c.metric(label, value)
    st.write("")
    a, b = st.columns([1.35, 1], gap="large")
    with a:
        st.subheader("From goals to outcomes")
        rh, ra = match_rates(
            fitted(model_name, trees, depth), snapshot, "Germany", "Scotland"
        )
        grid = goal_grid(rh, ra, 6)
        chart(
            px.imshow(
                grid * 100,
                x=list(range(7)),
                y=list(range(7)),
                labels={
                    "x": "Scotland goals",
                    "y": "Germany goals",
                    "color": "Probability (%)",
                },
                color_continuous_scale=["#f7f5ef", "#b4c0e5", "#2846a6"],
                title="Opening fixture · Germany vs Scotland",
                aspect="auto",
            ),
            height=350,
        )
        st.caption(
            f"Expected goals: Germany {rh:.2f}, Scotland {ra:.2f}. The displayed 0–6 grid contains {grid.sum():.2%} of score probability."
        )
    with b:
        st.subheader("Follow the analysis")
        st.write(
            "Start with the source records and lagged features. Compare the regression models, check the held-out predictions, then explore a match or run the whole tournament."
        )
        st.markdown("""1. **Data & features** — coverage, rankings and what was known before kickoff.
2. **Models & validation** — the baseline, errors and probability calibration.
3. **Match & tournament** — scorelines, group qualification and knockout paths.
4. **Presentation** — the findings and a downloadable research report.""")
        note(
            "This is a retrospective reconstruction, not a preserved forecast issued in June 2024. Later corrections to historical source records may be present. Actual EURO results are kept out of training."
        )
    with st.expander("What changed from the notebooks?"):
        st.write(
            "The same four rank-weighted inputs feed separate home and away regressors. Rankings now join backwards on their publication date. Model comparison uses 2023, followed by a separate 2024 holdout. Neutral fixtures average both team orderings. The tournament samples scorelines instead of assigning every match its most likely result."
        )
        st.write(
            "Despite its filename, “04. Gamma Model.ipynb” fits linear, Random Forest and gradient boosting regressors. The Poisson notebook loads the saved Random Forest models. No Gamma regression or XGBoost model is used here."
        )


def data_page():
    header(
        "02",
        "Start with the records.",
        "International results and FIFA ranking releases are the inputs. The bundled files let the app run without a Kaggle account or a live data connection.",
    )
    a, b, c = st.columns(3)
    a.metric("Source result rows", f"{len(matches):,}")
    b.metric("Ranking releases", ranks.rank_date.nunique())
    c.metric("Excluded from modelling", f"{len(features) - len(frame):,}")
    note(
        "Modelling uses results before 14 June 2024. Rows before August 2019 provide form history; teams need at least six earlier matches and a published ranking. Later records are used only on the tournament review page."
    )
    a, b = st.columns(2)
    with a:
        annual = (
            frame.groupby(frame.date.dt.year)
            .size()
            .rename("Matches")
            .reset_index(name="Matches")
        )
        chart(
            px.bar(
                annual,
                x="date",
                y="Matches",
                title="Model-ready matches by year",
                color_discrete_sequence=COLORS,
            ),
            height=330,
        )
    with b:
        goals = (
            pd.concat([frame.home_score, frame.away_score])
            .value_counts()
            .sort_index()
            .rename_axis("Goals")
            .reset_index(name="Team performances")
        )
        chart(
            px.bar(
                goals,
                x="Goals",
                y="Team performances",
                title="Recorded goals per team",
                color_discrete_sequence=[COLORS[1]],
            ),
            height=330,
        )
    team = st.selectbox("Inspect a team", TEAMS, index=TEAMS.index("England"))
    tm = frame[(frame.home_team == team) | (frame.away_team == team)]
    st.dataframe(
        tm[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "rank_home",
                "rank_away",
                "rank_date_home",
                "rank_date_away",
                "tournament",
            ]
        ].sort_values("date", ascending=False),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download model-ready match data",
        frame.to_csv(index=False),
        "euro-model-ready.csv",
        "text/csv",
    )
    with st.expander("Sources, hashes and target definition"):
        st.markdown(
            "[International results — Mart Jürisoo](https://github.com/martj42/international_results) · [Original FIFA ranking dataset — cashncarry](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)"
        )
        st.write(
            "The ranking file is an April 2024 snapshot mirrored in the source listed below. Exact commits and file hashes are recorded for reproducibility. These are men’s international matches."
        )
        st.write(
            "The results source includes extra-time goals, excludes penalty-shootout goals and does not identify every extra-time match. The regressions therefore target recorded match goals. Reusing their rates for 90-minute Poisson forecasts is an approximation. Tournament outcome scoring below uses only the 36 group games, which have no extra time."
        )
        st.json(json.loads((ROOT / "data/manifest.json").read_text()))


def feature_page():
    header(
        "03",
        "What the model knows before kickoff.",
        "Recent results describe scoring, conceding and points won. FIFA ranking differences weight those summaries, following the feature selection in the original exploratory notebook.",
    )
    st.latex(r"d=r_{home}-r_{away},\qquad x_{goals}=\overline{g}_{past}\,d")
    st.write(
        "A lower FIFA rank is better, so a negative difference favours the first team. Each model uses four inputs. Home goals use the home attack and away defence; away goals use the away attack and home defence. Both include each team’s average match points, multiplied by the same rank difference."
    )
    defs = pd.DataFrame(
        {
            "Input": [
                "Attack × rank difference",
                "Opponent defence × rank difference",
                "Home points × rank difference",
                "Away points × rank difference",
            ],
            "Source": [
                "Mean goals scored before this date",
                "Mean goals conceded before this date",
                "Mean 3/1/0 match points before this date",
                "Mean 3/1/0 match points before this date",
            ],
        }
    )
    st.dataframe(defs, hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        corr = frame[HOME_FEATURES + ["home_score"]].corr()
        chart(
            px.imshow(
                corr,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale="RdBu",
                title="Home model · feature correlations",
                aspect="auto",
            ),
            height=380,
        )
    with b:
        team = st.selectbox("Team form history", TEAMS, index=TEAMS.index("England"))
        h = features[features.home_team == team][
            ["date", "home_goals", "home_goals_l5"]
        ].rename(
            columns={
                "home_goals": "All earlier games",
                "home_goals_l5": "Last five games",
            }
        )
        a = features[features.away_team == team][
            ["date", "away_goals", "away_goals_l5"]
        ].rename(
            columns={
                "away_goals": "All earlier games",
                "away_goals_l5": "Last five games",
            }
        )
        form = pd.concat([h, a]).sort_values("date")
        chart(
            px.line(
                form,
                x="date",
                y=["All earlier games", "Last five games"],
                title=f"{team} · lagged scoring averages",
                labels={"value": "Goals per match", "date": "Match date"},
            ),
            height=380,
        )
    note(
        "The last-five series is shown for comparison; the four-input notebook models use expanding averages. Neither the current score nor another result from the same date can enter its features."
    )
    st.subheader("The assumptions worth checking")
    st.write(
        "Multiplying every input by the rank difference makes all four zero when ranks are equal. Correlated form summaries can also make feature importance unstable. Form since 2017 may over-weight an older squad. These are useful reasons to compare the model with a simple baseline before adding more complexity."
    )
    st.dataframe(snapshot.reset_index(), hide_index=True, width="stretch")


def model_page():
    header(
        "04",
        "Fit the goals. Compare the errors.",
        "Each method fits two regressors, one for each side’s goal total. The validation set is the full 2023 calendar year; smaller errors are better.",
    )
    table = comparison(trees, depth)
    st.dataframe(
        table.style.format({c: "{:.3f}" for c in table.columns if c != "Model"}),
        hide_index=True,
        width="stretch",
    )
    a, b = st.columns(2)
    with a:
        chart(
            px.bar(
                table.sort_values("Goal RMSE", ascending=False),
                x="Goal RMSE",
                y="Model",
                orientation="h",
                title="2023 validation · goal RMSE",
                color_discrete_sequence=COLORS,
            ),
            height=330,
        )
    with b:
        st.subheader("What each method adds")
        st.write(
            "The mean baseline predicts the average goals in the fitting period. Linear regression tests whether a weighted sum is enough. Random Forest averages decision trees. Gradient boosting builds small trees in sequence to correct earlier errors."
        )
        st.write(
            f"The selected model is **{model_name}**. Change it in the sidebar. Random Forest settings are applied only when you submit the form; every estimator uses a fixed random seed."
        )
        st.caption(
            "Linear predictions and tree outputs are clipped to 0.05–6 goals before probability calculations. This is a modelling guardrail, not an estimated confidence interval."
        )
    models = fitted(model_name, trees, depth)
    if hasattr(models[0], "feature_importances_"):
        imp = pd.DataFrame(
            {"Feature": HOME_FEATURES, "Importance": models[0].feature_importances_}
        ).sort_values("Importance")
        chart(
            px.bar(
                imp,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Home regressor · fitted split importance",
                color_discrete_sequence=[COLORS[2]],
            ),
            height=300,
        )
        st.caption(
            "Split importance describes this fitted model. It is not a causal effect and can favour features with more possible split points."
        )
    with st.expander("Training windows and model settings", expanded=False):
        st.markdown("""- **Validation:** fit August 2019–December 2022; score January–December 2023.
- **Holdout:** refit through December 2023; score January–13 June 2024.
- **Tournament:** refit through 13 June 2024; freeze team form before the opening game.""")
        st.code(
            json.dumps(
                {
                    "model": model_name,
                    "Random Forest": {
                        "n_estimators": trees,
                        "max_depth": depth,
                        "min_samples_leaf": 5,
                        "random_state": 42,
                    },
                    "Gradient boosting": {
                        "n_estimators": 100,
                        "max_depth": 2,
                        "learning_rate": 0.05,
                        "random_state": 42,
                    },
                },
                indent=2,
            ),
            language="json",
        )


def validation_page():
    header(
        "05",
        "Check the predictions on later matches.",
        "These matches are from January to 13 June 2024. Their scores did not enter the fitted holdout model. Form features can use earlier results within this period, just as they would in a rolling match forecast.",
    )
    test, metrics = validation(model_name, trees, depth)
    cols = st.columns(4)
    for c, (label, value) in zip(cols, metrics.items()):
        c.metric(label, f"{value:.3f}")
    st.caption(
        f"{len(test):,} held-out matches · 1X2 Brier is the sum across three outcome classes (range 0–2). Repeatedly choosing settings on this page would turn it into another validation set."
    )
    long = pd.concat(
        [
            pd.DataFrame(
                {"Actual": test.home_score, "Predicted": test.pred_home, "Side": "Home"}
            ),
            pd.DataFrame(
                {"Actual": test.away_score, "Predicted": test.pred_away, "Side": "Away"}
            ),
        ]
    )
    long["Residual"] = long.Actual - long.Predicted
    a, b = st.columns(2)
    with a:
        chart(
            px.scatter(
                long,
                x="Predicted",
                y="Actual",
                color="Side",
                opacity=0.4,
                title="Predicted vs recorded goals",
                color_discrete_sequence=COLORS,
            ),
            height=340,
        )
    with b:
        chart(
            px.histogram(
                long,
                x="Residual",
                color="Side",
                barmode="overlay",
                opacity=0.6,
                nbins=24,
                title="Goal residuals · actual minus predicted",
                color_discrete_sequence=COLORS,
            ),
            height=340,
        )
    actual_home = (test.home_score > test.away_score).astype(int)
    bins = pd.cut(test.p_home, bins=np.linspace(0, 1, 6), include_lowest=True)
    cal = (
        pd.DataFrame({"Bin": bins, "Predicted": test.p_home, "Observed": actual_home})
        .groupby("Bin", observed=True)
        .agg(
            Predicted=("Predicted", "mean"),
            Observed=("Observed", "mean"),
            Matches=("Observed", "size"),
        )
        .reset_index()
    )
    fig = px.scatter(
        cal,
        x="Predicted",
        y="Observed",
        size="Matches",
        hover_data=["Matches"],
        title="Home-win reliability · five fixed probability bins",
        range_x=[0, 1],
        range_y=[0, 1],
    )
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="#9b9c9a", dash="dash")
    )
    chart(fig, height=340)
    st.caption(
        "A point on the diagonal has the same predicted and observed win rate. Bin size matters: a few matches can produce a large apparent gap. These intervals are not a calibration guarantee."
    )
    test["Absolute error"] = (
        abs(test.home_score - test.pred_home) + abs(test.away_score - test.pred_away)
    ) / 2
    st.dataframe(
        test[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "pred_home",
                "pred_away",
                "p_home",
                "p_draw",
                "p_away",
                "Absolute error",
            ]
        ].sort_values("Absolute error", ascending=False),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Download holdout predictions",
        test.to_csv(index=False),
        "euro-holdout-predictions.csv",
        "text/csv",
    )


def match_page():
    header(
        "06",
        "One match, a range of scorelines.",
        "The regressors produce goal rates. Independent Poisson distributions turn those rates into a score grid; adding its cells gives win, draw and loss probabilities.",
    )
    a, b = st.columns(2)
    with a:
        home = st.selectbox("First team", TEAMS, index=TEAMS.index("England"))
    others = [t for t in TEAMS if t != home]
    with b:
        away = st.selectbox(
            "Second team",
            others,
            index=others.index("Spain") if "Spain" in others else 0,
        )
    rh, ra = rates(model_name, trees, depth)[home, away]
    with st.expander("Change the goal-rate assumptions"):
        multiplier = st.slider(
            "First-team scoring-rate multiplier", 0.5, 1.5, 1.0, 0.05
        )
        multiplier_a = st.slider(
            "Second-team scoring-rate multiplier", 0.5, 1.5, 1.0, 0.05
        )
        st.caption(
            "A sensitivity scenario. It changes the Poisson rates without retraining the regressors."
        )
    rh *= multiplier
    ra *= multiplier_a
    probs = outcome_probs(rh, ra)
    cols = st.columns(3)
    for c, label, p in zip(cols, [f"{home} win", "Draw", f"{away} win"], probs):
        c.metric(label, f"{p:.1%}")
    a, b = st.columns([1.3, 1])
    with a:
        grid = goal_grid(rh, ra, 8)
        chart(
            px.imshow(
                grid * 100,
                x=list(range(9)),
                y=list(range(9)),
                text_auto=".1f",
                labels={
                    "x": f"{away} goals",
                    "y": f"{home} goals",
                    "color": "Probability (%)",
                },
                title=f"{home} vs {away} · score probability (%)",
                color_continuous_scale=["#f7f5ef", "#a8b7df", "#2846a6"],
                aspect="auto",
            ),
            height=410,
        )
        st.caption(
            f"Visible grid mass: {grid.sum():.3%}. Headline outcome probabilities include the full Poisson tails, rather than renormalising this cropped grid."
        )
    with b:
        g = np.arange(10)
        df = pd.DataFrame(
            {"Goals": g, home: poisson.pmf(g, rh), away: poisson.pmf(g, ra)}
        )
        chart(
            px.bar(
                df,
                x="Goals",
                y=[home, away],
                barmode="group",
                title="Marginal goal distributions",
                labels={"value": "Probability"},
            ),
            height=330,
        )
        st.write(
            f"Goal rates: **{rh:.2f}** for {home}, **{ra:.2f}** for {away}. These are expected goal counts from match-level data, rather than shot-based xG."
        )
    note(
        "EURO fixtures are neutral-ground comparisons: both administrative team orderings are averaged. Goals are assumed independent. There is no lineup, injury, tactical or red-card model, so the percentages leave out information a football analyst would use."
    )
    scores = pd.DataFrame(
        [(i, j, float(grid[i, j])) for i in range(9) for j in range(9)],
        columns=[home, away, "Probability"],
    ).sort_values("Probability", ascending=False)
    st.dataframe(scores.head(8), hide_index=True, width="stretch")
    st.download_button(
        "Download score probabilities",
        scores.to_csv(index=False),
        "euro-score-grid.csv",
        "text/csv",
    )


def tournament_page():
    header(
        "07",
        "Play the tournament more than once.",
        "Each run samples all 36 group scorelines, qualifies 16 teams and plays the knockout bracket. The final probabilities count how often each team reaches a stage.",
    )
    with st.form("tournament_form"):
        a, b = st.columns(2)
        n = a.select_slider(
            "Tournament runs", options=[250, 500, 1000, 2500, 5000], value=1000
        )
        seed = int(
            b.number_input("Random seed", min_value=0, max_value=999999, value=2024)
        )
        run = st.form_submit_button("Run tournament", type="primary")
    if run:
        st.session_state["simulation_config"] = (n, seed)
    n, seed = st.session_state.get("simulation_config", (1000, 2024))
    table, points, bracket = tournament(model_name, trees, depth, n, seed)
    st.caption(
        f"{n:,} tournaments · seed {seed} · {model_name} · all rates frozen before 14 June 2024"
    )
    a, b = st.columns([1.15, 1])
    with a:
        top = table.reset_index().head(12).sort_values("Champion")
        fig = px.bar(
            top,
            x="Champion",
            y="Team",
            orientation="h",
            error_x=top["MC standard error"] * 1.96,
            title="Chance of winning · top 12",
            color_discrete_sequence=COLORS,
        )
        fig.update_xaxes(tickformat=".0%")
        chart(fig, height=460)
    with b:
        display = table.reset_index()[
            ["Team", "Champion", "Final", "Semi-final", "Quarter-final", "Round of 16"]
        ]
        st.dataframe(
            display.style.format({c: "{:.1%}" for c in display.columns if c != "Team"}),
            hide_index=True,
            width="stretch",
            height=460,
        )
    st.caption(
        "Whiskers show roughly ±1.96 Monte Carlo standard errors. They measure simulation noise only; model uncertainty is much larger and is not included."
    )
    a, b = st.columns(2)
    with a:
        team = st.selectbox(
            "Group-points distribution", TEAMS, index=TEAMS.index("England")
        )
        chart(
            px.bar(
                x=list(points.columns),
                y=points.loc[team].to_numpy(),
                labels={"x": "Group points", "y": "Probability"},
                title=f"{team} · points after three matches",
                color_discrete_sequence=[COLORS[2]],
            ),
            height=340,
        )
    with b:
        st.subheader("One sampled knockout bracket")
        st.dataframe(bracket, hide_index=True, width="stretch", height=340)
        st.caption(
            "The final sampled run is shown as an example. It is not the most likely bracket."
        )
    with st.expander("The six groups"):
        st.dataframe(
            pd.DataFrame(GROUPS)
            .T.rename(columns={i: f"Team {i + 1}" for i in range(4)})
            .rename_axis("Group")
            .reset_index(),
            hide_index=True,
            width="stretch",
        )
    with st.expander("Tournament mechanics and simplifications"):
        st.write(
            "The six groups and the 15 possible third-place allocations follow EURO 2024. Group ties use head-to-head points, goal difference and goals, reapplying the mini-table to unresolved subsets, then overall goal difference and goals. Third-place teams use points, goal difference, goals and wins."
        )
        st.write(
            "Disciplinary totals, qualifying-rank tiebreakers and the special last-match shootout are not modelled; unresolved ties use a seeded draw. Knockout draws add 30 minutes at one-third of the goal rates, then a 50/50 shootout if still level. Form and rankings do not update during a simulated tournament."
        )
        st.markdown(
            "[UEFA 2024 qualification and third-place explanation](https://www.uefa.com/uefaeuro/history/news/028e-1b3221d55b32-0a95528a80d5-1000--uefa-euro-2024-best-third-placed-teams/)"
        )
    st.download_button(
        "Download stage probabilities",
        table.to_csv(),
        "euro-stage-probabilities.csv",
        "text/csv",
    )


def presentation_page():
    header(
        "08",
        "The results, ready to discuss.",
        "A short research summary with the forecast, held-out performance and the assumptions that affect how much confidence to place in it. Download it as a standalone HTML report.",
    )
    n, seed = st.session_state.get("simulation_config", (1000, 2024))
    table, _, _ = tournament(model_name, trees, depth, n, seed)
    test, metrics = validation(model_name, trees, depth)
    leader = table.index[0]
    prob = table.iloc[0].Champion
    st.subheader(f"{leader} leads this model’s forecast at {prob:.1%}.")
    st.write(
        f"That still leaves a {1 - prob:.1%} chance of somebody else winning. The forecast uses {model_name.lower()}, with team form frozen before the opening match and {n:,} simulated tournaments at seed {seed}."
    )
    top = table.reset_index().head(10).sort_values("Champion")
    fig = px.bar(
        top,
        x="Champion",
        y="Team",
        orientation="h",
        title="Pre-tournament model · championship probability",
        color_discrete_sequence=COLORS,
    )
    fig.update_xaxes(tickformat=".0%")
    chart(fig, height=400)
    st.subheader("How did it look against the actual group games?")
    euros = matches[
        (matches.tournament == "UEFA Euro")
        & (matches.date >= "2024-06-14")
        & (matches.date <= "2024-06-26")
    ].copy()
    matrix = rates(model_name, trees, depth)
    predictions = np.array(
        [matrix[h, a] for h, a in zip(euros.home_team, euros.away_team)]
    )
    euros[["pred_home", "pred_away"]] = predictions
    euro_metrics = score_metrics(
        euros[["home_score", "away_score"]].to_numpy(), predictions
    )
    a, b, c = st.columns(3)
    a.metric("Group games reviewed", len(euros))
    b.metric("Group-game Brier", f"{euro_metrics['1X2 Brier']:.3f}")
    c.metric("Pre-EURO holdout RMSE", f"{metrics['Goal RMSE']:.3f}")
    st.caption(
        "Actual group results are retrospective evaluation only. Their inputs use one frozen pre-tournament snapshot. This differs from the rolling historical holdout and neither is an untouched contemporary forecast archive."
    )
    st.dataframe(
        euros[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "pred_home",
                "pred_away",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    note(
        "The main limitations are the independence assumption, rank-weighted form features, incomplete separation of extra-time goals in the training source, and simplified tournament tiebreaks. More simulated runs reduce sampling noise; they do not fix these assumptions."
    )
    report = build_report(
        table,
        test,
        metrics,
        euros,
        euro_metrics,
        fig,
        model_name,
        trees,
        depth,
        n,
        seed,
    )
    st.download_button(
        "Download research report (.html)",
        report,
        "euro-2024-research-report.html",
        "text/html",
        type="primary",
    )
    st.download_button(
        "Download actual group-game review (.csv)",
        euros.to_csv(index=False),
        "euro-2024-group-review.csv",
        "text/csv",
    )


pages = [
    st.Page(overview, title="Overview", default=True),
    st.Page(data_page, title="Data & coverage", url_path="data"),
    st.Page(feature_page, title="Feature engineering", url_path="features"),
    st.Page(model_page, title="Model bench", url_path="models"),
    st.Page(validation_page, title="Validation", url_path="validation"),
    st.Page(match_page, title="Match lab", url_path="match"),
    st.Page(tournament_page, title="Tournament", url_path="tournament"),
    st.Page(presentation_page, title="Presentation", url_path="presentation"),
]
st.navigation(pages).run()
with st.sidebar:
    st.divider()
    st.markdown(
        "[Original notebooks & source](https://github.com/steveaq/ML-2024-Euros-Model)"
    )
    st.caption("Python · scikit-learn · SciPy · Plotly")
