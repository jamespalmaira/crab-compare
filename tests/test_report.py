"""report.html chart builder (no GPU, no results folder).

    python -m unittest discover -s tests -t . -v
"""
import math
import unittest

import pandas as pd

from src.report import _nice_max, bar_chart, training_section


class ReportTests(unittest.TestCase):
    def test_bar_chart_has_one_bar_and_tooltip_per_model(self):
        svg = bar_chart("Macro-F1", ["A", "B", "C"], [0.2, 0.9, 0.0], [0.05, float("nan"), None], x_max=1.0)
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>"))
        self.assertEqual(svg.count("<title>"), 3)
        self.assertEqual(svg.count('fill="var(--series-1)"'), 2)  # a zero value draws no bar
        self.assertIn("A: 0.200 ± 0.050", svg)
        self.assertIn("B: 0.900<", svg)  # NaN std: no whisker or ± in the tooltip

    def test_labels_are_escaped(self):
        svg = bar_chart("t <x>", ["<b>&"], [1.0])
        self.assertNotIn("<b>&", svg)
        self.assertIn("&lt;b&gt;&amp;", svg)

    def test_training_section_counts_and_flags(self):
        per_seed = pd.DataFrame({
            "attribute": ["genus"] * 6, "arch": ["a"] * 3 + ["b"] * 3, "model": ["A"] * 3 + ["B"] * 3,
            "epochs_trained": [50, 20, 30, 15, 16, 50], "best_epoch": [45, 10, 20, 5, 6, 40]})
        html = "".join(training_section(per_seed, 50, ["b", "a"]))
        self.assertLess(html.index("<td>B</td>"), html.index("<td>A</td>"))  # model order kept
        self.assertIn("33.3 (20–50)", html)
        self.assertIn("25.0 (10–45)", html)
        self.assertEqual(html.count("1 of 3"), 2)
        self.assertIn("Flagged (a third or more of runs used all 50 epochs): B, A", html)
        per_seed["epochs_trained"] = [20, 20, 30, 15, 16, 49]
        self.assertIn("No model had", "".join(training_section(per_seed, 50)))

    def test_nice_axis_max(self):
        self.assertEqual(_nice_max(403.4), 500)
        self.assertEqual(_nice_max(0.37), 0.5)
        self.assertEqual(_nice_max(8.9), 10)
        self.assertTrue(math.isclose(_nice_max(0.0), 1.0))


if __name__ == "__main__":
    unittest.main()
