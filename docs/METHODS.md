# Reconstructing the six notebooks

The notebooks remain the historical source. The Streamlit application is a maintained reconstruction, with reproducibility and time-aware evaluation added. Exact original model outputs cannot be reproduced: their local input files and saved pickles were not committed.

| Notebook | Application |
| --- | --- |
| 01. Database Creator | Bundled public results and ranking snapshots; backward as-of ranking joins |
| 02. Feature Engineering | Expanding pre-match scoring, conceding and match-points averages; lagged last-five diagnostics |
| 03. Exploratory Data Analysis | Coverage, goal histograms, feature correlations and selected four-feature dictionaries |
| 04. Gamma Model | Separate home/away linear regression, Random Forest and gradient boosting; mean baseline |
| 05. Poisson Model | Independent goal distributions, complete outcome tails and sampled tournament |
| 06. Clean Output | Interactive match presentation and downloadable HTML research report |

The Gamma filename is historical: its code does not fit Gamma regression. It saves Random Forest estimators, despite also comparing gradient boosting. XGBoost is not part of these notebooks.

## Information dates

Only results before 14 June 2024 enter features and final model fitting. Results from 2017 onward provide form history; model fitting starts in August 2019, with at least six previous games for each team. This extra warm-up differs from the notebook's shorter history. FIFA releases are joined backwards by team and original publication date, never by a date rewritten to today. All games on a date are featurised before any scores from that date enter history. Rankings published on match day are assumed available that day; publication times are not present in the source.

The 2023 validation comparison fits through December 2022. The 2024 holdout refits through December 2023. Form can update after earlier held-out matches; model coefficients cannot. The final tournament model fits all eligible pre-tournament matches and freezes form before kickoff. The 36 actual EURO group games remain a separate retrospective evaluation.

These are reconstructed historical records downloaded later. Source corrections may post-date 2024, so this is not an archived contemporary backtest.

## Goal model and probability layer

The notebook's four inputs multiply mean scoring, opponent conceding and each team's average match points by the FIFA rank difference. Home and away targets are fitted separately. Neutral tournament forecasts average predictions from both administrative orderings, removing an arbitrary label effect. Outputs are clipped to 0.05–6 goals for every estimator; this is an explicit guardrail, not a learned bound.

Independent Poisson marginals define score probabilities. The displayed score grid is cropped for readability; 1X2 probabilities use the full Skellam difference distribution, so tail probability is preserved. Brier scores sum the squared errors across the three classes and range from zero to two. Log loss uses natural logarithms. Goal MAE/RMSE aggregate both teams equally.

The source scores can include extra time and do not include shootout goals. No reliable extra-time flag exists throughout the historical source. Applying these goal rates to 90-minute tournament games is therefore an approximation. The app states this next to data definitions and presentation results.

## Simulation

The six actual EURO groups feed a round-robin simulation. The six group winners, runners-up and four best third-place teams enter the official 15-case third-place allocation and knockout bracket. Group ranking uses head-to-head points, goal difference and goals, reapplied to unresolved subsets; overall goal difference and goals follow. Third-place comparisons use points, goal difference, goals and wins.

Disciplinary and qualifier-ranking tiebreakers, and the exceptional final-group-game shootout, are not modelled. Unresolved ties use seeded draws. Knockout draws add 30 minutes at one-third rates, followed by a 50/50 penalty shootout if needed. No form, fatigue or strength update occurs during a run. Displayed Monte Carlo error bars describe sampling error, not parameter or structural uncertainty.

Reference: [UEFA 2022–24 regulations, Articles 20–22](https://tavovidal.com/docs/2024%20Regulations%20Euro%20Cup.pdf) and [UEFA third-place explanation](https://www.uefa.com/uefaeuro/history/news/028e-1b3221d55b32-0a95528a80d5-1000--uefa-euro-2024-best-third-placed-teams/).

## Further work

Useful extensions include a 90-minute-only training source, time-decayed form, explicit neutral/home advantage, opponent-adjusted strengths, a direct Poisson regression baseline, calibration fitted on validation data and rolling-origin validation over several tournaments. Each should be measured against the existing baseline before it is used in a tournament forecast.
