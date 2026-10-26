"""Dataset builder tests: mocked API responses, no network, no image downloads.

    python -m unittest discover -s tools/dataset_builder/tests -t . -v
"""
import io
import json
import logging
import os
import re
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw

from tools.dataset_builder import build, dedupe, gbif, grouping, inat, labels, licenses, manual, outputs
from tools.dataset_builder.cache import QueryCache
from tools.dataset_builder.download import DownloadLog, materialize, verify_image
from tools.dataset_builder.net import Client, MissingContact, contact_email
from tools.dataset_builder.records import new_record
from tools.dataset_builder.tests.fakes import (GBIF, INAT, TERMS, FakeClient, FakeResponse, default_routes, gbif_occ,
                                               inat_obs, load_cfg)

CFG = load_cfg()
TH = CFG["thresholds"]


def rec(**kw):
    base = dict(source="inaturalist", species="Scylla serrata", photo_id=f"p{id(kw)}")
    base.update(kw)
    return new_record(**base)


def textured(seed, size=128):
    """A random grayscale texture: distinct seeds give distinct perceptual hashes."""
    rng = np.random.default_rng(seed)
    small = (rng.random((8, 8)) * 255).astype(np.uint8)
    return Image.fromarray(small).resize((size, size), Image.BICUBIC).convert("RGB")


class LicenseTests(unittest.TestCase):
    def test_normalize_inat_gbif_and_free_text(self):
        cases = {"cc-by-nc": "CC-BY-NC", "cc0": "CC0", "cc-by-nc-sa": "CC-BY-NC-SA",
                 "http://creativecommons.org/licenses/by-nc/4.0/legalcode": "CC-BY-NC",
                 "https://creativecommons.org/licenses/by-sa/4.0/": "CC-BY-SA",
                 "http://creativecommons.org/publicdomain/zero/1.0/": "CC0",
                 "CC BY 4.0": "CC-BY", "CC-BY-4.0": "CC-BY", "CC0: Public Domain": "CC0",
                 "cc-by-nd": "CC-BY-ND", "© Jane, all rights reserved": licenses.ARR,
                 None: "", "": "", "MIT": licenses.OTHER}
        for raw, code in cases.items():
            self.assertEqual(licenses.normalize(raw), code, raw)

    def test_allowlist(self):
        allow = CFG["licenses"]["allowlist"]
        self.assertEqual(licenses.check("cc-by-nc-sa", allow), ("CC-BY-NC-SA", ""))
        self.assertEqual(licenses.check(None, allow)[1], "license_missing")
        self.assertEqual(licenses.check("all rights reserved", allow)[1], "license_all_rights_reserved")
        self.assertEqual(licenses.check("cc-by-nd", allow)[1], "license_not_allowed:CC-BY-ND")
        self.assertEqual(licenses.check("cc-by-nc-nd", allow)[1], "license_not_allowed:CC-BY-NC-ND")
        self.assertEqual(licenses.check("MIT", allow)[1], "license_not_allowed:OTHER")
        self.assertEqual(licenses.check("MIT", allow + ["MIT"])[1], "")


class LabelTests(unittest.TestCase):
    def test_controlled_terms_found_by_label(self):
        from tools.dataset_builder.tests.fakes import CONTROLLED_TERMS
        terms = labels.inat_terms(CONTROLLED_TERMS)
        self.assertEqual(terms["sex"][0], 9)
        self.assertEqual(terms["maturity"][0], 1)
        self.assertEqual(set(terms), {"sex", "maturity"})

    def test_inat_annotations(self):
        maps = CFG["labels"]
        female = [{"controlled_attribute_id": 9, "controlled_value_id": 11, "vote_score": 1}]
        self.assertEqual(labels.inat_annotation(female, TERMS["sex"], maps["sex"]), ("female", "Female"))
        undetermined = [{"controlled_attribute_id": 9, "controlled_value_id": 20, "vote_score": 0}]
        self.assertEqual(labels.inat_annotation(undetermined, TERMS["sex"], maps["sex"]), ("", "Cannot Be Determined"))
        voted_down = [{"controlled_attribute_id": 9, "controlled_value_id": 10, "vote_score": -1}]
        self.assertEqual(labels.inat_annotation(voted_down, TERMS["sex"], maps["sex"]), ("", ""))
        conflict = female + [{"controlled_attribute_id": 9, "controlled_value_id": 10, "vote_score": 0}]
        self.assertEqual(labels.inat_annotation(conflict, TERMS["sex"], maps["sex"])[0], "")
        larva = [{"controlled_attribute_id": 1, "controlled_value_id": 6, "vote_score": 0}]
        self.assertEqual(labels.inat_annotation(larva, TERMS["maturity"], maps["maturity"]), ("", "Larva"))
        juvenile = [{"controlled_attribute_id": 1, "controlled_value_id": 8}]
        self.assertEqual(labels.inat_annotation(juvenile, TERMS["maturity"], maps["maturity"])[0], "juvenile")
        self.assertEqual(labels.inat_annotation(female, None, maps["sex"]), ("", ""))

    def test_inat_photo_records_carry_labels_and_sources(self):
        o = inat_obs(5, 77, [(501, "cc-by"), (502, None)],
                     annotations=[{"controlled_attribute_id": 9, "controlled_value_id": 10, "vote_score": 0},
                                  {"controlled_attribute_id": 1, "controlled_value_id": 2, "vote_score": 0}])
        rs = inat.photo_records(o, "Scylla serrata", "PH", TERMS, CFG)
        self.assertEqual(len(rs), 2)
        r = rs[0]
        self.assertEqual((r["sex"], r["sex_source"], r["maturity"], r["maturity_source"]),
                         ("male", "inaturalist:annotation", "adult", "inaturalist:annotation"))
        self.assertEqual(r["image_url"], "https://inaturalist-open-data.s3.amazonaws.com/photos/501/medium.jpg")
        self.assertEqual((r["observer_id"], r["observation_id"]), ("inaturalist:77", "inaturalist:5"))
        self.assertEqual((r["latitude"], r["longitude"]), (14.5, 121.0))
        self.assertEqual((r["width"], r["height"]), (2048, 1536))

    def test_obscured_location_is_not_published(self):
        o = inat_obs(6, 1, [(601, "cc0")], obscured=True)
        r = inat.photo_records(o, "Scylla serrata", "PH", TERMS, CFG)[0]
        self.assertEqual((r["latitude"], r["longitude"], r["location_note"]), ("", "", "obscured"))

    def test_gbif_sex_and_life_stage(self):
        o = gbif_occ(9, [("https://x.org/a.jpg", "cc-by")], sex="FEMALE", lifeStage="Juvenile")
        r = gbif.photo_records(o, "Scylla serrata", CFG)[0]
        self.assertEqual((r["sex"], r["sex_source"], r["maturity"], r["maturity_source"]),
                         ("female", "gbif:sex", "juvenile", "gbif:lifeStage"))
        o = gbif_occ(10, [("https://x.org/b.jpg", "cc-by")], sex="HERMAPHRODITE", lifeStage="Egg")
        r = gbif.photo_records(o, "Scylla serrata", CFG)[0]
        self.assertEqual((r["sex"], r["sex_source"], r["sex_raw"]), ("", "", "HERMAPHRODITE"))
        self.assertEqual((r["maturity"], r["maturity_raw"]), ("", "Egg"))

    def test_disease_column_is_always_empty(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            outputs.write_labels([rec(image_path="a.jpg", crab_id="observer:x")], tmp / "labels.csv")
            df = pd.read_csv(tmp / "labels.csv", dtype=str, keep_default_na=False)
            self.assertEqual(list(df.columns), list(outputs.LABEL_COLUMNS))
            self.assertEqual(df.loc[0, "disease"], "")
        finally:
            shutil.rmtree(tmp)


class DuplicateTests(unittest.TestCase):
    def test_gbif_copy_of_inat_observation_is_dropped(self):
        inat_recs = inat.photo_records(inat_obs(100, 1, [(1001, "cc-by")]), "Scylla serrata", "PH", TERMS, CFG)
        dup = gbif_occ(1, [("https://inaturalist-open-data.s3.amazonaws.com/photos/1001/original.jpg", "cc-by")],
                       inat_obs_id=100)
        only_gbif = gbif_occ(2, [("https://inaturalist-open-data.s3.amazonaws.com/photos/2002/original.jpg", "cc-by")],
                             inat_obs_id=200)
        museum = gbif_occ(3, [("https://museum.org/x.jpg", "http://creativecommons.org/licenses/by/4.0/")])
        gbif_recs = [r for o in (dup, only_gbif, museum) for r in gbif.photo_records(o, "Scylla serrata", CFG)]
        records = inat_recs + gbif_recs
        n = dedupe.cross_source(records, CFG["source_priority"])
        self.assertEqual(n, 1)
        self.assertFalse(inat_recs[0]["skipped"])  # iNaturalist copy kept
        self.assertEqual(gbif_recs[0]["skip_reason"], "duplicate_of_inaturalist")
        self.assertFalse(gbif_recs[1]["skipped"])  # iNaturalist-derived, but its iNaturalist copy is not in this run
        self.assertEqual(gbif_recs[1]["observation_id"], "inaturalist:200")
        self.assertFalse(gbif_recs[2]["skipped"])

    def test_inat_detection_from_references_or_dataset(self):
        key = CFG["gbif"]["inaturalist_dataset_key"]
        self.assertEqual(gbif.inat_observation_id({"references": "https://www.inaturalist.org/observations/42"}, key), "42")
        self.assertEqual(gbif.inat_observation_id({"datasetKey": key, "catalogNumber": "43"}, key), "43")
        self.assertEqual(gbif.inat_observation_id({"datasetKey": "x", "catalogNumber": "44"}, key), "")

    def test_gbif_dup_of_license_skipped_inat_photo_is_still_a_duplicate(self):
        inat_recs = inat.photo_records(inat_obs(100, 1, [(1001, None)]), "Scylla serrata", "PH", TERMS, CFG)
        g = gbif.photo_records(gbif_occ(1, [("https://inaturalist-open-data.s3.amazonaws.com/photos/1001/o.jpg",
                                             "cc-by")], inat_obs_id=100), "Scylla serrata", CFG)
        dedupe.cross_source(inat_recs + g, CFG["source_priority"])
        self.assertEqual(g[0]["skip_reason"], "duplicate_of_inaturalist")

    def test_repeated_url_is_dropped_once(self):
        a = rec(source="gbif", photo_id="gbif:1:0", image_url="https://x.org/same.jpg")
        b = rec(source="gbif", photo_id="gbif:2:0", image_url="https://x.org/same.jpg")
        dedupe.cross_source([a, b], CFG["source_priority"])
        self.assertEqual((a["skipped"], b["skip_reason"]), (False, "duplicate_photo"))


class NearDuplicateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_near_duplicates_removed_and_logged(self):
        base = textured(1)
        base.save(self.tmp / "a.png")
        base.resize((96, 96)).save(self.tmp / "a_small.jpg", quality=70)  # same photo, resized and recompressed
        textured(2).save(self.tmp / "b.png")
        recs = [rec(source="manual_set", photo_id="m:a_small", local_path=str(self.tmp / "a_small.jpg")),
                rec(source="inaturalist", photo_id="i:a", local_path=str(self.tmp / "a.png")),
                rec(source="gbif", photo_id="g:b", local_path=str(self.tmp / "b.png"))]
        for r in recs:
            r["phash"] = dedupe.phash(r["local_path"])
        log = dedupe.near_duplicates(recs, CFG["near_duplicates"]["max_distance"], CFG["source_priority"])
        self.assertEqual(len(log), 1)
        self.assertEqual((log[0]["removed_photo_id"], log[0]["kept_photo_id"]), ("m:a_small", "i:a"))
        self.assertEqual(recs[0]["skip_reason"], "near_duplicate")  # iNaturalist preferred over manual import
        self.assertFalse(recs[2]["skipped"])

    def test_crop_box_is_hashed_on_the_crop(self):
        im = Image.new("RGB", (200, 100), "white")
        im.paste(textured(3, 100), (100, 0))
        im.save(self.tmp / "two.png")
        textured(3, 100).save(self.tmp / "crop.png")
        self.assertEqual(dedupe.phash(self.tmp / "two.png", json.dumps([100, 0, 200, 100])),
                         dedupe.phash(self.tmp / "crop.png"))


class GroupingTests(unittest.TestCase):
    def test_levels_and_fallback(self):
        r = rec(observer_id="inaturalist:7", observation_id="inaturalist:70", photo_id="inaturalist:photo:700",
                file_id="crab146:12")
        self.assertEqual(grouping.group_key(r, "observer"), ("observer:inaturalist:7", "observer_id"))
        self.assertEqual(grouping.group_key(r, "observation"), ("observation:inaturalist:70", "observation_id"))
        self.assertEqual(grouping.group_key(r, "file_id"), ("file:crab146:12", "file_id"))
        bare = rec(observation_id="gbif:5", photo_id="gbif:5:0")
        self.assertEqual(grouping.group_key(bare, "observer"), ("observation:gbif:5", "observation_id"))
        self.assertEqual(grouping.group_key(bare, "file_id"), ("observation:gbif:5", "observation_id"))
        self.assertEqual(grouping.group_key(rec(photo_id="x:1"), "observation"), ("photo:x:1", "photo_id"))

    def test_same_observer_shares_crab_id(self):
        rs = (inat.photo_records(inat_obs(1, 9, [(11, "cc0"), (12, "cc0")]), "Scylla serrata", "PH", TERMS, CFG)
              + inat.photo_records(inat_obs(2, 9, [(21, "cc0")]), "Scylla serrata", "PH", TERMS, CFG))
        grouping.assign(rs, "observer")
        self.assertEqual(len({r["crab_id"] for r in rs}), 1)
        grouping.assign(rs, "observation")
        self.assertEqual(len({r["crab_id"] for r in rs}), 2)


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def spec(self, **kw):
        s = {"name": "set1", "path": "set1", "license": "CC-BY-4.0", "url": "https://example.org/set1",
             "attribution": "Example authors"}
        s.update(kw)
        p = self.tmp / f"{s['name']}.yaml"
        p.write_text(json.dumps(s), encoding="utf-8")  # JSON is valid YAML
        return manual.load_spec(p, self.tmp)

    def test_filename_regex(self):
        rx = re.compile(r"^(?P<id>\d+)_")
        self.assertEqual(manual.parse_file_id(rx, "012_left.jpg"), "012")
        self.assertEqual(manual.parse_file_id(rx, "left_012.jpg"), "")
        self.assertEqual(manual.parse_file_id(re.compile(r"crab(\d+)"), "img_crab7_a.png"), "7")
        self.assertEqual(manual.parse_file_id(re.compile(r"[A-Z]\d+"), "x_B12.png"), "B12")

    def test_filename_id_import(self):
        d = self.tmp / "set1" / "mud"
        d.mkdir(parents=True)
        for name in ("001_a.jpg", "001_b.jpg", "002_a.jpg", "nolabel.jpg"):
            textured(len(name)).save(d / name)
        spec = self.spec(type="filename_id", regex=r"^(?P<id>\d+)_", class_map={"mud": "Scylla serrata"})
        rs = manual.run_import(spec, CFG)
        kept = [r for r in rs if not r["skipped"]]
        self.assertEqual(sorted(r["file_id"] for r in kept), ["set1:001", "set1:001", "set1:002"])
        self.assertEqual([r["skip_reason"] for r in rs if r["skipped"]], ["file_id_not_parsed"])
        grouping.assign(kept, "file_id")
        self.assertEqual(len({r["crab_id"] for r in kept}), 2)
        self.assertTrue(all(r["species"] == "Scylla serrata" and r["source"] == "set1" for r in kept))

    def test_folder_csv_import(self):
        d = self.tmp / "set1"
        d.mkdir()
        textured(1).save(d / "a.jpg")
        textured(2).save(d / "b.jpg")
        (d / "labels.csv").write_text("file,species,sex,observer_id\na.jpg,Scylla serrata,Female,obs1\n"
                                      "b.jpg,,unknown,\n", encoding="utf-8")
        rs = manual.run_import(self.spec(type="folder_csv", species="Portunus pelagicus"), CFG)
        self.assertEqual([r["species"] for r in rs], ["Scylla serrata", "Portunus pelagicus"])
        self.assertEqual((rs[0]["sex"], rs[0]["sex_source"], rs[0]["observer_id"]), ("female", "manual:set1:csv", "set1:obs1"))
        self.assertEqual((rs[1]["sex"], rs[1]["sex_raw"]), ("", "unknown"))
        self.assertEqual((rs[0]["license_raw"], rs[0]["observation_url"]), ("CC-BY-4.0", "https://example.org/set1"))

    def test_spec_requires_license_and_url(self):
        p = self.tmp / "bad.yaml"
        p.write_text("name: bad\ntype: folder_csv\npath: x\nurl: https://x\n", encoding="utf-8")
        with self.assertRaises(manual.ImportSpecError):
            manual.load_spec(p, self.tmp)
        p.write_text("name: gbif\ntype: folder_csv\npath: x\nurl: u\nlicense: CC0\n", encoding="utf-8")
        with self.assertRaises(manual.ImportSpecError):
            manual.load_spec(p, self.tmp)

    def test_roboflow_yolo_crops_one_record_per_box(self):
        base = self.tmp / "set1" / "train"
        (base / "images").mkdir(parents=True)
        (base / "labels").mkdir()
        (self.tmp / "set1" / "data.yaml").write_text("names: [mud_crab, crab, blue_female]\n", encoding="utf-8")
        Image.new("RGB", (200, 100), "gray").save(base / "images" / "img1_jpg.rf.abc123.jpg")
        (base / "labels" / "img1_jpg.rf.abc123.txt").write_text(
            "0 0.25 0.5 0.4 0.8\n1 0.75 0.5 0.4 0.8\n2 0.6 0.4 0.7 0.4 0.7 0.6 0.6 0.6\n0 0.5 0.5 0.05 0.05\n",
            encoding="utf-8")
        spec = self.spec(type="roboflow_yolo", class_map={"mud_crab": "Scylla serrata",
                                                          "blue_female": {"species": "Portunus pelagicus", "sex": "female"}})
        rs = manual.run_import(spec, CFG)
        self.assertEqual(len(rs), 4)
        self.assertEqual(json.loads(rs[0]["crop_box"]), [10, 10, 90, 90])
        self.assertEqual(rs[1]["skip_reason"], "class_not_mapped:crab")
        self.assertEqual((rs[2]["species"], rs[2]["sex"], rs[2]["sex_source"]),
                         ("Portunus pelagicus", "female", "manual:set1:class_map"))
        self.assertEqual(json.loads(rs[2]["crop_box"]), [120, 40, 140, 60])
        self.assertEqual(rs[2]["skip_reason"], "box_too_small")  # 20 px < 32
        self.assertEqual(rs[3]["skip_reason"], "box_too_small")
        self.assertEqual(rs[0]["observation_id"], "set1:img1")  # Roboflow augmented copies share the source image

    def test_roboflow_coco(self):
        d = self.tmp / "set1" / "valid"
        d.mkdir(parents=True)
        Image.new("RGB", (100, 100), "gray").save(d / "x.jpg")
        coco = {"images": [{"id": 1, "file_name": "x.jpg", "width": 100, "height": 100}],
                "categories": [{"id": 0, "name": "crabs"}, {"id": 1, "name": "mud_crab"}],
                "annotations": [{"image_id": 1, "category_id": 1, "bbox": [10, 20, 50, 60]}]}
        (d / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
        rs = manual.run_import(self.spec(type="roboflow_coco", class_map={"mud_crab": "Scylla serrata"}), CFG)
        self.assertEqual(len(rs), 1)
        self.assertEqual(json.loads(rs[0]["crop_box"]), [10, 20, 60, 80])
        self.assertEqual((rs[0]["width"], rs[0]["height"]), (50, 60))
        self.assertFalse(rs[0]["skipped"])


class CountsTests(unittest.TestCase):
    def make(self):
        recs = []
        # Scylla serrata: 31 observers, 2 photos each, all inaturalist; sex on observers 0-30 alternating
        for u in range(31):
            for k in range(2):
                recs.append(rec(photo_id=f"i:{u}:{k}", observer_id=f"inaturalist:{u}", observation_id=f"inaturalist:{u}",
                                license="CC-BY", country="PH", sex="female" if u % 2 else "male",
                                sex_source="inaturalist:annotation"))
        # Portunus pelagicus: 3 observers, mixed sources
        for u in range(3):
            recs.append(rec(species="Portunus pelagicus", source="gbif" if u else "inaturalist", photo_id=f"g:{u}",
                            observer_id=f"gbif:{u}", observation_id=f"gbif:{u}", license="CC0", country="ID",
                            maturity="adult", maturity_source="gbif:lifeStage"))
        skipped = rec(photo_id="s1", observer_id="inaturalist:99", license="ARR")
        skipped.update(skipped=True, skip_reason="license_all_rights_reserved")
        dup = rec(source="gbif", photo_id="s2", observer_id="gbif:x")
        dup.update(skipped=True, skip_reason="duplicate_of_inaturalist")
        return recs + [skipped, dup]

    def test_counts_table(self):
        counts = outputs.counts_table(self.make(), ["Scylla serrata", "Portunus pelagicus", "Charybdis feriata"], TH,
                                      capped={"Portunus pelagicus"}).set_index("species")
        s = counts.loc["Scylla serrata"]
        self.assertEqual((s["photos"], s["groups_observer"], s["groups_observation"]), (62, 31, 31))
        self.assertEqual((s["sex_labeled"], s["maturity_labeled"]), (62, 0))
        self.assertEqual((s["skipped_license"], s["skipped_duplicate"]), (1, 1))
        self.assertTrue(s["trainable"])
        self.assertTrue(s["dominated"])
        self.assertEqual(json.loads(s["photos_by_license"]), {"CC-BY": 62})
        p = counts.loc["Portunus pelagicus"]
        self.assertEqual((p["photos"], p["groups_observer"], p["trainable"], p["cap_hit"]), (3, 3, False, True))
        self.assertEqual(json.loads(p["photos_by_source"]), {"gbif": 2, "inaturalist": 1})
        self.assertAlmostEqual(p["top_source_share"], 0.667)
        self.assertFalse(p["dominated"])
        c = counts.loc["Charybdis feriata"]
        self.assertEqual((c["photos"], c["groups_observer"], c["trainable"]), (0, 0, False))
        self.assertEqual(counts.loc["ALL", "photos"], 65)

    def test_attribute_verdicts(self):
        attrs = outputs.attribute_table(self.make(), TH)
        verdicts = outputs.attribute_verdicts(attrs, TH)
        sex = attrs[attrs["attribute"] == "sex"].set_index("class")
        self.assertEqual((sex.loc["female", "groups_observer"], sex.loc["male", "groups_observer"]), (15, 16))
        self.assertFalse(verdicts["sex"][0])  # two classes, but each below 30 groups
        self.assertFalse(verdicts["maturity"][0])  # one class
        self.assertIn("1 class", verdicts["maturity"][1])
        many = []
        for u in range(60):
            many.append(rec(photo_id=f"x{u}", observer_id=f"o{u}", sex="female" if u < 30 else "male"))
        self.assertTrue(outputs.attribute_verdicts(outputs.attribute_table(many, TH), TH)["sex"][0])

    def test_format_counts_prints_breakdowns(self):
        text = outputs.format_counts(outputs.counts_table(self.make(), ["Scylla serrata"], TH))
        self.assertIn("groups_observer", text)
        self.assertIn('photos_by_source:', text)


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_resolve_taxa_and_place(self):
        client = FakeClient(default_routes())
        ok = inat.resolve_taxon(client, INAT, "Scylla serrata")
        self.assertEqual((ok["inat_status"], ok["inat_taxon_id"]), ("exact", 47185))
        bad = inat.resolve_taxon(client, INAT, "Scylla fakeus")
        self.assertEqual(bad["inat_status"], "higher_rank")
        self.assertEqual(gbif.match_species(client, GBIF, "Scylla serrata")["gbif_status"], "exact")
        self.assertEqual(gbif.match_species(client, GBIF, "Scylla fakeus")["gbif_status"], "higher_rank")
        self.assertEqual(inat.resolve_place(client, INAT, "Philippines"), (6888, "Philippines"))

    def test_inat_synonym(self):
        client = FakeClient({"/taxa": {"results": [{"id": 5, "name": "Portunus armatus", "rank": "species",
                                                    "matched_term": "Portunus pelagicus"}]}})
        r = inat.resolve_taxon(client, INAT, "Portunus pelagicus")
        self.assertEqual((r["inat_status"], r["inat_name"]), ("synonym", "Portunus armatus"))

    def test_inat_pagination_and_resume(self):
        cfg = json.loads(json.dumps(CFG))
        cfg["inaturalist"]["per_page"] = 2
        obs = [inat_obs(i, i, [(i * 10, "cc-by")]) for i in range(1, 6)]
        client = FakeClient(default_routes(observations=obs))
        cache = QueryCache(self.tmp)
        got, state = inat.fetch_observations(client, cfg, cache, "k", 47185, 6888, "research", limit=3)
        self.assertEqual((len(got), state["done"], state["capped"]), (4, False, True))  # stopped after 2 pages
        got, state = inat.fetch_observations(client, cfg, cache, "k", 47185, 6888, "research")
        self.assertEqual(([o["id"] for o in got], state["done"], state["total"]), ([1, 2, 3, 4, 5], True, 5))
        ids_above = [p["id_above"] for url, p in client.calls if url.endswith("/observations")]
        self.assertEqual(ids_above, [0, 2, 4])  # resumed from id 4, not from the start
        n = len(client.calls)
        inat.fetch_observations(client, cfg, cache, "k", 47185, 6888, "research")
        self.assertEqual(len(client.calls), n)  # finished queries come from the cache
        self.assertEqual(client.calls[0][1]["quality_grade"], "research")

    def test_gbif_pagination(self):
        cfg = json.loads(json.dumps(CFG))
        cfg["gbif"]["page_size"] = 2
        occs = [gbif_occ(k, [(f"https://x.org/{k}.jpg", "cc-by")]) for k in range(5)]
        client = FakeClient(default_routes(occurrences=occs))
        got, state = gbif.fetch_occurrences(client, cfg, QueryCache(self.tmp), "g", 2225776, "PH")
        self.assertEqual((len(got), state["done"]), (5, True))
        p = client.calls[0][1]
        self.assertEqual((p["mediaType"], p["country"], p["taxonKey"]), ("StillImage", "PH", 2225776))

    def test_cache_ignores_half_written_line(self):
        cache = QueryCache(self.tmp)
        cache.append("k", [{"id": 1}, {"id": 2}], {"id_above": 2})
        with open(self.tmp / "k.jsonl", "a", encoding="utf-8") as f:
            f.write('{"id": 2}\n{"id": 3, "trunc')
        recs, state = cache.load("k", "id")
        self.assertEqual([r["id"] for r in recs], [1, 2])
        self.assertEqual(state, {"id_above": 2})


class ClientTests(unittest.TestCase):
    def test_contact_email_required(self):
        with mock.patch.dict(os.environ, {"CRAB_TEST_EMAIL": "  "}):
            with self.assertRaises(MissingContact):
                contact_email("CRAB_TEST_EMAIL")
        with mock.patch.dict(os.environ, {"CRAB_TEST_EMAIL": "someone@example.org"}):
            self.assertEqual(contact_email("CRAB_TEST_EMAIL"), "someone@example.org")

    def test_backoff_and_rate_limit(self):
        session = mock.Mock()
        session.headers = {}
        busy = FakeResponse(status=429)
        busy.headers = {"Retry-After": "7"}
        session.get.side_effect = [requests.ConnectionError("down"), FakeResponse(status=503), busy,
                                   FakeResponse({"ok": 1})]
        sleeps, now = [], [0.0]
        client = Client("ua-test", {"api.example.org": 1.0}, session=session, sleep=sleeps.append,
                        clock=lambda: now[0], backoff_base=2.0)
        self.assertEqual(client.get_json("https://api.example.org/x"), {"ok": 1})
        self.assertEqual(session.headers["User-Agent"], "ua-test")
        backoffs = [s for s in sleeps if s != 1.0]
        self.assertEqual(backoffs, [2.0, 4.0, 8.0])  # Retry-After 7 < 8
        self.assertIn(1.0, sleeps)  # rate limit between requests (clock never advances)

    def test_gives_up(self):
        session = mock.Mock()
        session.headers = {}
        session.get.return_value = FakeResponse(status=500)
        client = Client("ua", session=session, sleep=lambda s: None, max_retries=3)
        with self.assertRaises(RuntimeError):
            client.get("https://x.org/")
        self.assertEqual(session.get.call_count, 3)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_invalid_image_rejected_and_valid_kept(self):
        buf = io.BytesIO()
        textured(1).save(buf, format="JPEG")
        client = mock.Mock()
        client.get.side_effect = [FakeResponse(content=b"<html>not an image</html>"), FakeResponse(content=buf.getvalue())]
        dlog = DownloadLog(self.tmp / "d.jsonl")
        bad = rec(photo_id="a", image_url="https://x.org/a.jpg", image_path="external/a.jpg")
        good = rec(photo_id="b", image_url="https://x.org/b.jpg", image_path="external/b.jpg")
        self.assertEqual(materialize(bad, self.tmp, client, dlog), (False, "invalid_image"))
        self.assertFalse((self.tmp / "external" / "a.jpg").exists())
        self.assertEqual(materialize(good, self.tmp, client, dlog), (True, ""))
        self.assertEqual(verify_image(self.tmp / "external" / "b.jpg"), (128, 128))
        # resumed: already-downloaded files are not fetched again
        self.assertEqual(materialize(good, self.tmp, client, DownloadLog(self.tmp / "d.jsonl")), (True, ""))
        self.assertEqual(client.get.call_count, 2)


class DryRunTests(unittest.TestCase):
    """End to end with mocked APIs: dry run writes metadata and counts but no image files."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        logging.disable(logging.WARNING)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        logging.disable(logging.NOTSET)

    def test_dry_run(self):
        species = self.tmp / "species.yaml"
        species.write_text("species:\n  - Scylla serrata\n  - Scylla fakeus\n", encoding="utf-8")
        sex = [{"controlled_attribute_id": 9, "controlled_value_id": 11, "vote_score": 0}]
        obs = [inat_obs(1, 1, [(11, "cc-by"), (12, "cc-by-nc")], annotations=sex),
               inat_obs(2, 1, [(21, None)]),
               inat_obs(3, 2, [(31, "cc0")])]
        occs = [gbif_occ(100, [("https://inaturalist-open-data.s3.amazonaws.com/photos/31/original.jpg", "cc0")],
                         inat_obs_id=3),
                gbif_occ(101, [("https://museum.org/a.jpg", "http://creativecommons.org/licenses/by/4.0/"),
                               ("https://museum.org/b.jpg", "© all rights reserved")], sex="MALE")]
        client = FakeClient(default_routes(obs, occs))
        out = io.StringIO()
        with redirect_stdout(out):
            rc = build.main(["--dry-run", "--species-file", str(species), "--data-dir", str(self.tmp / "data"),
                             "--imports"], client=client)
        self.assertEqual(rc, 0)
        run = self.tmp / "data" / "external" / "runs" / "default"
        for name in ("metadata.csv", "labels.csv", "counts.csv", "attribute_counts.csv", "ATTRIBUTION.md",
                     "taxa.csv", "places.csv", "warnings.txt", "run.json", "counts.txt"):
            self.assertTrue((run / name).exists(), name)
        self.assertFalse(any(p.suffix in (".jpg", ".png") for p in (self.tmp / "data").rglob("*")))
        self.assertFalse((self.tmp / "data" / "labels.csv").exists())
        self.assertFalse(any(url.startswith("https://inaturalist-open-data") for url, _ in client.calls))

        counts = pd.read_csv(run / "counts.csv").set_index("species")
        s = counts.loc["Scylla serrata"]
        self.assertEqual((s["photos"], s["groups_observer"], s["groups_observation"]), (4, 3, 3))
        self.assertEqual((s["skipped_license"], s["skipped_duplicate"], s["sex_labeled"]), (2, 1, 3))
        self.assertEqual(counts.loc["Scylla fakeus", "photos"], 0)
        labels_df = pd.read_csv(run / "labels.csv", dtype=str, keep_default_na=False)
        self.assertEqual(list(labels_df.columns), list(outputs.LABEL_COLUMNS))
        self.assertEqual(set(labels_df["source"]), {"inaturalist", "gbif"})
        self.assertTrue(labels_df["image_path"].str.startswith("external/").all())
        self.assertEqual(labels_df.loc[labels_df["sex"] == "male", "sex_source"].tolist(), ["gbif:sex"])
        warnings = (run / "warnings.txt").read_text(encoding="utf-8")
        self.assertIn("Scylla fakeus: iNaturalist higher_rank", warnings)
        self.assertIn("zero photos", warnings)
        place_calls = [p for url, p in client.calls if url.endswith("/observations")]
        self.assertTrue(all(p["place_id"] == 6888 for p in place_calls))  # looked up, not hardcoded
        self.assertIn("Counts per species", out.getvalue())

    def test_download_mode_with_mocked_images(self):
        species = self.tmp / "species.yaml"
        species.write_text("species:\n  - Scylla serrata\n", encoding="utf-8")
        obs = [inat_obs(1, 1, [(11, "cc-by"), (12, "cc-by")]), inat_obs(2, 2, [(21, "cc0")])]
        occs = [gbif_occ(100, [("https://museum.org/a.jpg", "cc-by"), ("https://museum.org/bad.jpg", "cc-by")])]
        images = {"/photos/11/medium.jpg": textured(1), "/photos/12/medium.jpg": textured(1).resize((100, 100)),
                  "/photos/21/medium.jpg": textured(2), "museum.org/a.jpg": textured(3)}

        def get(url, params=None):
            for suffix, im in images.items():
                if url.endswith(suffix):
                    buf = io.BytesIO()
                    im.save(buf, format="JPEG")
                    return FakeResponse(content=buf.getvalue())
            return FakeResponse(content=b"not an image")

        client = FakeClient(default_routes(obs, occs))
        client.get = get
        with redirect_stdout(io.StringIO()):
            rc = build.main(["--species-file", str(species), "--data-dir", str(self.tmp / "data"), "--imports"],
                            client=client)
        self.assertEqual(rc, 0)
        data = self.tmp / "data"
        labels_df = pd.read_csv(data / "labels.csv", dtype=str, keep_default_na=False)
        self.assertEqual(len(labels_df), 3)  # 12 is a near-duplicate of 11; bad.jpg is not an image
        self.assertTrue(all((data / p).exists() for p in labels_df["image_path"]))
        meta = pd.read_csv(data / "external" / "runs" / "default" / "metadata.csv", dtype=str, keep_default_na=False)
        reasons = dict(zip(meta["photo_id"], meta["skip_reason"]))
        self.assertEqual(reasons["inaturalist:photo:12"], "near_duplicate")
        self.assertEqual(reasons["gbif:100:1"], "invalid_image")
        self.assertTrue((data / "external" / "review" / "scylla_serrata.jpg").exists())
        near = pd.read_csv(data / "external" / "runs" / "default" / "near_duplicates.csv")
        self.assertEqual(near["removed_photo_id"].tolist(), ["inaturalist:photo:12"])

    def test_missing_contact_email_stops_before_network(self):
        with mock.patch.dict(os.environ, {"CRAB_DATASET_CONTACT_EMAIL": ""}):
            err = io.StringIO()
            with mock.patch("sys.stderr", err):
                rc = build.main(["--dry-run", "--data-dir", str(self.tmp / "data"), "--imports"])
        self.assertEqual(rc, 2)
        self.assertIn("CRAB_DATASET_CONTACT_EMAIL", err.getvalue())

    def test_countries_option(self):
        self.assertEqual(build.countries_from_arg("default", CFG), ["PH"])
        self.assertIn("ID", build.countries_from_arg("widened", CFG))
        self.assertEqual(build.countries_from_arg("ph,my", CFG), ["PH", "MY"])
        with self.assertRaises(ValueError):
            build.countries_from_arg("PH,XX", CFG)


if __name__ == "__main__":
    unittest.main()
