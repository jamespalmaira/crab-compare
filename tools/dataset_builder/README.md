# Dataset builder

Builds the crab image dataset for the bench from public sources only: iNaturalist,
GBIF, and datasets you download by hand. It is separate from the bench code in `src/`.

Labels come from the sources, never from the images. Species comes from the resolved
taxon. Sex and maturity come only from iNaturalist annotations and the GBIF `sex` and
`lifeStage` fields; they are weak and unverified. Disease is always empty, because no
public source provides it. `crab_id` is a grouping key (an observer, an observation, or
an ID from a filename), not a verified individual crab.

## Before you start

1. Confirm `species.yaml` with a taxonomist.
2. Set a contact email for the User-Agent. The tool reads it from the environment and
   never writes it to a file:

   ```bash
   export CRAB_DATASET_CONTACT_EMAIL=you@example.org        # PowerShell: $env:CRAB_DATASET_CONTACT_EMAIL = "you@example.org"
   ```

No API keys or logins are needed: every request is a public read. Requests go out at
about one per second to iNaturalist (0.5 s for GBIF), with backoff on errors.

## Dry run (metadata only, no images)

```bash
python -m tools.dataset_builder --dry-run                       # Philippines only
python -m tools.dataset_builder --dry-run --countries widened   # PH plus neighbouring countries (config.yaml)
python -m tools.dataset_builder --dry-run --countries PH,ID,MY
```

Options: `--quality-grade` (iNaturalist; default `research`, or `needs_id`, `casual`, a
comma list, or `any`), `--group-by observer|observation|file_id` (default `observer`, the
strictest), `--max-per-species N`, `--sources inaturalist,gbif`, `--imports FILE ...`,
`--run-name`.

Each run writes to `data/external/runs/<run-name>/` (not in git):

| File | Contents |
|---|---|
| `counts.txt` | the printed report: taxa, places, counts, thresholds, warnings |
| `counts.csv` | per species: photos, groups at each grouping level, photos per source, license and country, sex and maturity labels, skips (license, duplicates, near-duplicates, cap), source dominance, trainable flag |
| `attribute_counts.csv` | per sex and maturity class: photos, groups, source dominance |
| `metadata.csv` | every photo seen, kept or skipped, with URL, date, public location, quality grade, taxon, size, skip reason |
| `labels.csv` | kept photos in the bench's format |
| `ATTRIBUTION.md` | credit and license for every kept photo |
| `taxa.csv`, `places.csv` | how each name and country resolved |
| `warnings.txt`, `run.json` | threshold warnings; settings and totals of the run |

Raw API pages are cached in `data/external/_cache/` as they arrive, so an interrupted run
resumes where it stopped and a second run (for example widened after Philippines) only
fetches what is new. Delete the cache folder to refetch from scratch.

The thresholds in `config.yaml` (`thresholds`) decide the flags: a species is trainable
with at least 30 observer-level groups; an attribute with at least 2 classes, each with at
least 30 groups; a class is dominated when one source supplies more than 80% of its images.

## Download

Only after the counts have been reviewed and the download approved:

```bash
python -m tools.dataset_builder --countries PH
```

Images go to `data/external/<source>/<species>/`, each is checked to be a valid image,
near-duplicates are removed (perceptual hash), and the kept rows are written to
`data/labels.csv` (an existing file is kept unless `--overwrite-labels`). Contact sheets of
50 random images per species are written to `data/external/review/` for a visual check.
Downloads are logged in `downloads.jsonl`, so a rerun skips what is already on disk.

## Licenses and duplicates

- Only images whose license is on `licenses.allowlist` (CC0, CC-BY, CC-BY-SA, CC-BY-NC,
  CC-BY-NC-SA) are kept. All-rights-reserved, unlicensed, no-derivatives and unrecognised
  licenses are skipped and counted. For GBIF the image's own (media) license is used, not
  the record's.
- GBIF republishes iNaturalist observations. A GBIF record whose iNaturalist observation or
  photo is also in the iNaturalist results is dropped, keeping the iNaturalist copy.
- Near-duplicate images across all sources are removed by perceptual hash
  (`near_duplicates.max_distance`) and logged in `near_duplicates.csv`. In a dry run only
  images already on disk (manual imports) can be hashed.

## Manual imports

Roboflow, Kaggle, Zenodo and GitHub datasets need a login or a manual download, so the
tool never scrapes them. Download the dataset yourself into `data/manual/<name>/`, then
copy one of the examples in `imports/` to `imports/<name>.yaml` and fill in the source
name, license, URL and credit. Every `imports/*.yaml` is used by default.

| `type` | For | Notes |
|---|---|---|
| `folder_csv` | any folder of images plus a CSV | `file` column required; optional `species`, `sex`, `maturity`, `observer_id`, `observation_id`, `id`, `license` |
| `roboflow_yolo`, `roboflow_coco` | Roboflow exports | each box becomes one cropped image labeled through `class_map`; unmapped classes and boxes under `min_box_size` are skipped; Roboflow's augmented copies of one image share an observation group |
| `filename_id` | ReID-style sets such as Crab-146 | `regex` extracts the crab ID from each filename (use `--group-by file_id`); files that do not match are skipped |

Species names must match `species.yaml`; others are skipped and counted.

## Citing the sources

- **iNaturalist:** credit each photo as in `ATTRIBUTION.md` (photographer and license) and
  cite iNaturalist (https://www.inaturalist.org) with the access date. CC-BY-NC photos may
  be used for non-commercial research only; check that the dissertation's use fits.
- **GBIF:** GBIF asks that data be cited with a DOI. API searches do not create one, so
  register a derived dataset at https://www.gbif.org/derived-dataset listing the GBIF
  records used (from `metadata.csv`) and cite the DOI it gives. Credit each photo's rights
  holder as in `ATTRIBUTION.md`.
- **Manual imports:** cite each dataset as its authors ask (paper, DOI or repository), using
  the URL and credit from its YAML file.

## Tests

```bash
python -m unittest discover -s tools/dataset_builder/tests -t . -v
```

The tests use mocked API responses and make no network requests.
