# EURO 2024 model notes

The app reads like a research article with controls: quiet paper backgrounds, navy text and one blue accent. The Editorial skill from [awesome-design-skills](https://github.com/bergside/awesome-design-skills/tree/main/skills/editorial) supplied the hierarchy, spacing and accessibility guidance. Its MIT-licensed source is saved alongside this file.

Typography uses a local serif display fallback and Streamlit's system sans; network fonts are optional. All charts share the same colour sequence, background and labels. Each page has a numbered section, a descriptive heading and a short introduction. The sidebar keeps model settings available across the analysis.

Controls retain Streamlit's visible labels and keyboard behaviour. Buttons have a minimum height of 44 pixels. Custom focus outlines are visible; reduced-motion preferences disable decorative transitions. Chart colours are supplemented by labels, axes and tables. At narrow widths columns stack; navigation collapses into Streamlit's sidebar. Source data and predictions can be downloaded as CSV, with a standalone HTML report for sharing.

Checks: navigate all eight routes; change both match selectors; change model settings; rerun a seeded tournament; open downloaded report; inspect chart labels at mobile width; verify keyboard focus. No unsupported precision or fictional live data is presented.
