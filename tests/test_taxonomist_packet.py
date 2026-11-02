"""The taxonomist packet: a fixed, blind, per-genus audit sample and the licence rule for images.

    python -m unittest discover -s tests -t . -v
"""
import unittest

import pandas as pd

from scripts.taxonomist_packet import (AUDIT_COLUMNS, FLAGGED, audit_table, draw_audit_sample, flagged_table, questions_text,
                                       shareable)


def rows(counts, license_name="CC-BY-NC"):
    out = []
    for genus, n in counts.items():
        for i in range(n):
            out.append({"photo_id": f"{genus}:{i}", "genus": genus, "species": f"{genus} a", "crab_id": f"{genus}{i}",
                        "image_path": f"{genus}/{i}.jpg", "observation_url": f"https://x/{genus}/{i}", "country": "SG",
                        "license": license_name, "attribution": "(c) someone", "source": "inaturalist", "observer_id": "o"})
    return pd.DataFrame(out)


class AuditSampleTests(unittest.TestCase):
    def test_takes_40_per_genus_or_all_if_fewer(self):
        sample = draw_audit_sample(rows({"A": 100, "B": 40, "C": 7}))
        self.assertEqual(sample["genus"].value_counts().to_dict(), {"A": 40, "B": 40, "C": 7})
        self.assertFalse(sample["photo_id"].duplicated().any())

    def test_fixed_seed_gives_the_same_draw_whatever_the_row_order(self):
        df = rows({"A": 100, "B": 50})
        first = draw_audit_sample(df)["photo_id"].tolist()
        self.assertEqual(draw_audit_sample(df.sample(frac=1, random_state=3))["photo_id"].tolist(), first)
        self.assertNotEqual(draw_audit_sample(df, seed=1)["photo_id"].tolist(), first)

    def test_the_order_does_not_group_by_genus(self):
        genera = draw_audit_sample(rows({"A": 100, "B": 100}))["genus"].tolist()
        self.assertNotEqual(genera, sorted(genera))

    def test_audit_table_is_blind_with_empty_answer_columns(self):
        table = audit_table(draw_audit_sample(rows({"A": 45})))
        self.assertEqual(list(table.columns), AUDIT_COLUMNS)
        self.assertFalse({"genus", "species"} & set(table.columns))
        for col in ("taxonomist_label", "confidence", "notes"):
            self.assertTrue((table[col] == "").all())
        self.assertEqual(table["sheet"].value_counts().sort_index().tolist(), [30, 10])


class LicenceTests(unittest.TestCase):
    def test_only_licences_that_allow_sharing_get_images(self):
        for name in ("CC0", "CC-BY", "CC-BY-SA", "CC-BY-NC", "CC-BY-NC-SA"):
            self.assertTrue(shareable(name))
        for name in ("CC-BY-ND", "CC-BY-NC-ND", "all rights reserved", ""):
            self.assertFalse(shareable(name))

    def test_a_photo_that_cannot_be_shared_is_a_link_only(self):
        table = audit_table(draw_audit_sample(rows({"A": 3}, "CC-BY-ND")))
        self.assertTrue((table["sheet"] == "").all())
        self.assertTrue(table["image_included"].str.startswith("no").all())
        self.assertTrue(table["observation_url"].str.startswith("https://").all())


class FlaggedTests(unittest.TestCase):
    def frame(self):
        df = pd.DataFrame([{**rows({"Scylla": 1}).iloc[0].to_dict(), "photo_id": p, "image_path": f"{i}.jpg"}
                           for i, p in enumerate(FLAGGED)])
        return df

    def test_lists_all_ten_photos_with_a_reason(self):
        table = flagged_table(self.frame())
        self.assertEqual(len(table), 10)
        self.assertTrue((table["reason_flagged"].str.len() > 0).all())
        self.assertTrue(table["reason_flagged"].str.contains("not an expert identification").all())

    def test_a_missing_photo_stops_the_packet(self):
        with self.assertRaises(SystemExit):
            flagged_table(self.frame().iloc[1:])

    def test_questions_cover_every_topic(self):
        df = rows({"Scylla": 2})
        text = questions_text(df, draw_audit_sample(df), flagged_table(self.frame()))
        for phrase in ("Portunus pelagicus", "S. paramamosain", "Zosimus", "Lophozozymus", "It is not an edible crab",
                       "flagged.csv", "20261102"):
            self.assertIn(phrase, text)


if __name__ == "__main__":
    unittest.main()
