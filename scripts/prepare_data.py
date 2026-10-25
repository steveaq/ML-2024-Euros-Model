"""Prepare the bundled, date-frozen public data. Run from the repository root."""

from pathlib import Path
from hashlib import sha256
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SOURCES = {
    "results": "https://raw.githubusercontent.com/martj42/international_results/d5bd6de012c87e2f325de602ae77fe2f9ab61ef3/results.csv",
    "rankings": "https://raw.githubusercontent.com/Rohxn14/fifa-wc2026-prediction/a4db631299ebcd193b535978488b43ecdc032ceb/fifa_ranking-2024-04-04.csv",
}


def main():
    DATA.mkdir(exist_ok=True)
    manifest = {
        "forecast_cutoff": "2024-06-14",
        "ranking_ceiling": "2024-04-04",
        "sources": {},
    }
    for name, url in SOURCES.items():
        raw = DATA / f"{name}-source.csv"
        if not raw.exists():
            import urllib.request

            with urllib.request.urlopen(url, timeout=60) as r:
                raw.write_bytes(r.read())
        df = pd.read_csv(raw)
        if name == "results":
            df = df[(df.date >= "2017-01-01") & (df.date <= "2024-07-14")].dropna(
                subset=["home_score", "away_score"]
            )
            df = df[
                [
                    "date",
                    "home_team",
                    "away_team",
                    "home_score",
                    "away_score",
                    "tournament",
                    "neutral",
                ]
            ]
        else:
            df = df[(df.rank_date >= "2016-01-01") & (df.rank_date <= "2024-04-04")]
            df = df[["rank_date", "country_full", "rank", "total_points"]]
        target = DATA / f"{name}.csv.gz"
        df.to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
        manifest["sources"][name] = {
            "url": url,
            "raw_sha256": sha256(raw.read_bytes()).hexdigest(),
            "bundled_sha256": sha256(target.read_bytes()).hexdigest(),
            "rows": len(df),
        }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
