"""Render each real page with Streamlit's test runner."""

import unittest
import ast
from pathlib import Path
from streamlit.testing.v1 import AppTest


def page_source():
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
    boundary = next(
        n.lineno
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pages" for t in n.targets)
    )
    return "\n".join(source.splitlines()[: boundary - 1])


class AppSmokeTests(unittest.TestCase):
    def test_each_page_renders(self):
        source = page_source()
        for page in [
            "overview",
            "data_page",
            "feature_page",
            "model_page",
            "validation_page",
            "match_page",
            "tournament_page",
            "presentation_page",
        ]:
            with self.subTest(page=page):
                app = AppTest.from_string(
                    source + f"\n{page}()\n", default_timeout=90
                ).run()
                self.assertEqual([e.message for e in app.exception], [])
                self.assertEqual(len(app.title), 1)

    def test_tournament_submit_and_model_change(self):
        source = page_source()
        app = AppTest.from_string(
            source + "\ntournament_page()\n", default_timeout=90
        ).run()
        next(b for b in app.button if b.label == "Run tournament").click().run()
        self.assertEqual([e.message for e in app.exception], [])
        next(s for s in app.selectbox if s.key == "goal_model").select(
            "Linear regression"
        ).run()
        self.assertEqual([e.message for e in app.exception], [])
        self.assertEqual(app.session_state["simulation_config"], (1000, 2024))


if __name__ == "__main__":
    unittest.main()
