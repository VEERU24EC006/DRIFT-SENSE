# Drift-Sense: Navigation-Error Recovery

Drift-Sense localizes a periodic semiconductor reference pattern inside a
1000x1000 px scanning-electron-microscope (SEM) style search image despite
**10x scale differences**, rotation, contrast/gain drift, noise, and the
periodic-ambiguity problem that breaks naive template matching.

A fully synthetic, ground-truth-labeled dataset generator produces reference +
search pairs for **DRAM-style** layouts (periodic word-lines and bit-lines
crossing at right angles with a contact/via at every intersection). A classical
(no-deep-learning) multi-hypothesis matcher predicts the reference center
`(x, y)` in the search image.

---

## 1. Setup

Requires **Python 3.10+** and only two runtime packages
(`numpy`, `opencv-python`):

```bash
git clone <your-repo-url>
cd kla_drift_sense

# create a virtual environment (Windows shown; Linux/macOS: python3 -m venv .venv)
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt
```

On the frozen 30-sample benchmark the pipeline runs in ~3.6 s per pair on a
consumer CPU (4 OpenCV threads, 4 Python worker threads).

---

## 2. Generate a sample image pair

```bash
# DRAM-style layout, 30 pairs, output to ./verify_data (fresh dir - recommended
# for a first run so the shipped ./dataset benchmark is never touched)
python dataset_generator.py --pairs 30 --output verify_data
```

> The generator refuses to overwrite a non-empty output directory unless you
> pass `--force` (it DELETES the directory contents). Keep the frozen
> `./dataset` benchmark intact by always generating into a fresh directory.

Arguments:

| Argument | Meaning |
|---|---|
| `--pairs N` (alias `--samples N`) | Number of reference + search pairs (≥30 recommended) |
| `--output DIR` | Output directory (default `dataset`) |
| `--force` | Allow overwriting a non-empty output directory |

For every pair the generator writes:

```
output/
  search/sample_000_search.png   # 1000x1000 field of view
  reference/sample_000_ref.png   # 1000x1000 reference pattern
  metadata/sample_000_metadata.json
  manifest.csv                   # true center per pair (ground truth)
  manifest.json
```

**Ground truth** (`target_center_x`, `target_center_y` in `manifest.csv`) is the
exact parent-geometry center of the reference crop inside the search window —
not a re-measured estimate — so evaluation is deterministic.

### Noise & augmentation model (with citations)

| Degradation | Implementation | Citation |
|---|---|---|
| Beam blur (Gaussian PSF) | `GaussianBlur`, sigma scaled per image | Goldstein et al. [1] |
| Shot/Poisson-style detector noise | Gaussian approximation, σ 2–4 DN (ref) / 3–6 DN (search) | Foi et al. [2] |
| Search noisier than reference | larger detector-noise σ on the search image | [2], test-data rule |
| SEM edge brightening | unsharp-mask edge enhancement on both images | [1], [10] |
| Contrast/gain & offset drift | `x*N(0.92,1.08) + N(-6,6)` | [1], SEM metrology practice |
| Global gamma / detector response | power-law `x^(1/γ)`, γ∈(0.94,1.06) | [1], [7] |
| Scale difference (up to 10x) | Parent 11000px → search 1000px (INTER_AREA) | [4] |
| Rotation | −2..+2 deg, BORDER_REFLECT | registration practice [6] |
| Feature-size variation | critical dimensions ×{0.5,1.0,2.0}, pitch fixed | [8] |
| Local jitter / missing features / weak contacts | per-cell jitter, 2% dropouts | [8] |

---

## 3. Run localization inference

`evaluation_script.py` is the inference script: it accepts a reference image
path and a search image path and prints the predicted center.

```bash
# predict the center of reference.png inside search.png
python evaluation_script.py --reference path/to/reference.png --search path/to/search.png

# optional: evaluator-side beam blur (nm FWHM) and CSV result logging
python evaluation_script.py --reference r.png --search s.png --beam_spot_nm 10 --output results.csv
```

Output (final line is the coordinate):

```
RESULT: (269.3000, 712.9000)
```

It runs without manual edits and only requires `numpy` and `opencv-python`.
The same file drives the benchmark validator (`final_production_check.py`).

---

## 4. Algorithm overview

1. **Hypothesis grid** — 125 transforms of the reference: 5 feature-size
   factors (morphological erode/dilate of critical features), 5 scales
   (0.08–0.12, the 10x resampling direction), 5 rotations (±2°).
2. **Zero-variance guard** — OpenCV `TM_CCOEFF_NORMED` returns an all-ones map
   for zero-variance templates (`templNorm < eps` in OpenCV's
   `common_matchTemplate`); such hypotheses are mathematically invalid and are
   skipped before matching.
3. **Parallel normalized cross-correlation** — `cv2.matchTemplate` per
   hypothesis across a 4-thread pool; top-8 peaks each; spatially deduplicated
   into a 48-candidate pool.
4. **Census validation** — regional (3x3) Census transform (Zabih & Woodfill)
   re-scores the top-20 pool candidates; a candidate only displaces the NCC
   winner when it gains regionally, resolving periodic-lattice ambiguity.
5. **Adaptive safety gate** — a near-perfect (ZNCC ≥ 0.999999) Adaptive result
   that also beats the RAW baseline is pathological (e.g. impulse noise), so
   the RAW baseline coordinate is returned instead.

### Frozen 30-sample result

`python final_production_check.py` (30-case benchmark shipped in `benchmark.csv`,
`dataset/`):

| Metric | Value |
|---|---|
| ≤ 0.65 px of GT | 93.3% (28/30) |
| mean / median / worst error | 27.7 / 0.4 / 497.8 px |
| mean runtime per 1000x1000 pair | ≈ 3.6 s |

Two pre-existing failures (`sample_010`, `sample_025`) are highly repetitive
layouts where a correct-but-alternate lattice location is visually
indistinguishable; the matcher still returns a self-consistent periodic
location.

---

## 5. Repository layout

```
kla_drift_sense/
  dataset_generator.py      # standalone DRAM-style dataset generator (GT)
  evaluation_script.py      # INFERENCE script (hypotheses, NCC, census, gates)
  final_production_check.py # frozen-30 regression validator
  benchmark.csv             # frozen baseline results
  dataset/                  # frozen 30-case benchmark (never regenerate)
  references.md             # citations for all noise/augmentation choices
  requirements.txt
  README.md
```

## 6. License / citation

See `references.md` for the full citation list used to justify the
noise, degradation, and matching-method choices.
