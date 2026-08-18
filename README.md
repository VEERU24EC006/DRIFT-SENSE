# Drift-Sense: Navigation-Error Recovery

Localizes a periodic DRAM-style semiconductor reference pattern inside a 1000x1000 px SEM-style search image, handling the 10x scale difference, rotation, contrast/gain drift, detector noise, and periodic-lattice ambiguity that breaks naive template matching.

The repo contains a synthetic dataset generator (with exact ground truth), the localization inference script, a frozen 30-pair benchmark validator, and a visual-report generator.

## Setup

Requires Python 3.10+ and only `numpy` + `opencv-python`.

```
git clone https://github.com/VEERU24EC006/DRIFT-SENSE.git
cd DRIFT-SENSE
python -m venv .venv
```

Activate the virtual environment:

Windows PowerShell:
```
.venv\Scripts\Activate.ps1
```

Linux/macOS:
```
source .venv/bin/activate
```

Install dependencies:
```
pip install -r requirements.txt
```

## 1. Generate a dataset (reference + search pairs)

```
python dataset_generator.py --pairs 30 --output verify_data
```

Arguments:

| Argument | Meaning |
|---|---|
| `--pairs N` (alias `--samples N`) | Number of image pairs to generate (minimum 30 recommended) |
| `--output DIR` | Output directory (defaults to `dataset`) |
| `--force` | Overwrite a non-empty output directory (deletes its contents) |

Use a fresh directory (like `verify_data` above) so the shipped `dataset/` benchmark is never touched.

For every pair the generator writes:

```
verify_data/
  search/sample_000_search.png     # 1000x1000 field of view (the search image)
  reference/sample_000_ref.png     # 1000x1000 reference pattern to find
  metadata/sample_000_metadata.json
  manifest.csv                     # true center per pair (ground truth)
  manifest.json
```

Ground truth for each pair is the `target_center_x` and `target_center_y` columns in `manifest.csv` — the exact parent-geometry center of the reference crop inside the search window.

Each generated pair combines: Gaussian beam blur, independent detector noise per image (search is noisier than reference), SEM edge brightening, contrast/gain and gamma drift, rotation in +/-2 deg, 10x scale reduction, feature-size variation, local jitter, and a small fraction of missing/weak contacts. Every choice is cited in `references.md`.

## 2. Run localization inference

`evaluation_script.py` takes a reference image and a search image and prints the predicted center of the reference pattern inside the search image.

```
python evaluation_script.py --reference test_input\reference.png --search test_input\search.png
```

Output:

```
RESULT: (452.0000, 263.0000)
```

The `RESULT:` line is the coordinate to use. A `#` line below it shows the internal decision, ZNCC score, scale, rotation, feature-size factor, and runtime. Optional flags: `--beam_spot_nm` (evaluator-side beam blur) and `--output` (append a CSV row).

Running `python evaluation_script.py` with no arguments uses the same `test_input\reference.png` / `test_input\search.png` demo pair.

## 3. Validate on the frozen benchmark

The repo ships a frozen 30-pair benchmark (`dataset/` + `benchmark.csv`). Run the validator to reproduce the reported accuracy:

```
python final_production_check.py
```

It runs the current `evaluation_script.py` on all 30 pairs and reports the error distribution, per-pair runtime, and whether results differ from the frozen baseline.

## 4. Generate the result visuals (success / failure)

```
python dataset_visual_generator.py
```

Randomly picks one success case (error <= 0.65 px) and one failure case (error > 5 px) from the frozen benchmark and saves two figures:

```
slide_success.png   # reference + search + zoomed panel with predicted (green X) and true (red circle)
slide_failure.png   # same layout plus a written reason for the failure
```

Options: `--seed N` for a reproducible pick, or `--success sample_000 --failure sample_010` to force specific samples.

## 5. Inspecting the generated images

Open any pair directly — the files are normal PNGs:

- `test_input\search.png` — the 1000x1000 field of view
- `test_input\reference.png` — the 1000x1000 reference pattern

Generated datasets follow the same layout under `verify_data\search\` and `verify_data\reference\`.

To check a prediction against ground truth, run the inference on a pair and compare the `RESULT:` line with that sample's `target_center_x` / `target_center_y` row in `manifest.csv`. For the frozen benchmark the same check is done automatically by `final_production_check.py`.

## Results (frozen 30-sample benchmark)

| Metric | Value |
|---|---|
| Predictions <= 0.65 px of true center | 93.3% (28/30) |
| Mean / median / worst error | 27.7 / 0.4 / 497.8 px |
| Mean runtime per 1000x1000 pair | ~3.6 s (4 OpenCV threads, 4 worker threads) |

Two pre-existing failures (`sample_010`, `sample_025`) are highly repetitive layouts where a different lattice instance is visually indistinguishable from the true one; the matcher still returns a self-consistent periodic location.

## Algorithm

1. Build 125 transforms of the reference: 5 feature-size factors (morphological erode/dilate), 5 scales (0.08-0.12, covering the 10x resampling direction), 5 rotations (+/-2 deg).
2. Skip zero-variance templates — OpenCV's `TM_CCOEFF_NORMED` returns an all-ones map for them, so they are excluded before matching.
3. Run normalized cross-correlation per hypothesis in a 4-thread pool, take the top 8 peaks each, and spatially deduplicate into a 48-candidate pool.
4. Re-score the top-20 pool candidates with a regional (3x3) Census transform; a candidate replaces the NCC winner only when it gains regionally, resolving periodic-lattice ambiguity.
5. If the selected Adaptive candidate is a near-perfect (ZNCC >= 0.999999) match that also beats the RAW baseline — a pathological response under impulse noise — return the RAW baseline coordinate instead.

## Repository layout

```
dataset_generator.py      # synthetic DRAM-style dataset generator (writes GT)
evaluation_script.py      # localization inference script (reference + search -> RESULT)
final_production_check.py # frozen 30-sample benchmark validator
dataset_visual_generator.py # success/failure figure generator
benchmark.csv             # frozen baseline results
dataset/                  # frozen 30-case benchmark (do not regenerate)
references.md             # citations for every noise/augmentation choice
requirements.txt
README.md
```

## References

All noise, degradation, and matching-method choices are justified in `references.md`.