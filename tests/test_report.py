"""report.html chart builder (no GPU, no results folder).

    python -m unittest discover -s tests -t . -v
"""
import math
import unittest

from src.report import _nice_max, bar_chart


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

    def test_nice_axis_max(self):
        self.assertEqual(_nice_max(403.4), 500)
        self.assertEqual(_nice_max(0.37), 0.5)
        self.assertEqual(_nice_max(8.9), 10)
        self.assertTrue(math.isclose(_nice_max(0.0), 1.0))


if __name__ == "__main__":
    unittest.main()
