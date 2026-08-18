# DRIFT-SENSE

## AI-Powered Navigation-Error Recovery for DRAM-Style Wafer Inspection

## 1. Overview

**DRIFT-SENSE** is a classical computer-vision system for recovering the location of a target semiconductor site inside a large SEM search image.

The system receives:

- a **Reference Image** of `1000 × 1000` pixels
- a **Search Image** of `1000 × 1000` pixels

The reference represents the target structure at approximately **10× higher magnification** than the search image. DRIFT-SENSE resamples the reference to the search scale (roughly 0.08–0.12 of its size), searches for the corresponding structure, and returns its center coordinates:

```text
(x, y)
```

in **Search Image pixel coordinates**.

The system is designed for the **Drift-Sense: Navigation-Error Recovery** problem, where stage drift, imaging variation, rotation, scale differences, noise, charging, and other acquisition effects can cause the inspection tool to land at the wrong location. This is particularly difficult for periodic layouts such as DRAM, where an incorrect location may look extremely similar to the intended site.

## 2. Key Idea

A conventional template matcher can fail when the highest correlation peak belongs to another repeated structure.

DRIFT-SENSE therefore uses a **coarse-to-fine, multi-hypothesis localization pipeline**:

```text
Reference + Search
        ↓
Scale / Rotation / Feature-Size Hypotheses
        ↓
Normalized Cross-Correlation (NCC)
        ↓
Multiple Candidate Peaks
        ↓
Spatial Suppression + Candidate Deduplication
        ↓
3×3 Regional Census Verification
        ↓
Final Candidate Selection
        ↓
Predicted Center (x, y)
```

Instead of trusting one correlation maximum, the system retains multiple candidates and verifies local structure before producing the final location.

## 3. Architecture Choice

The primary target architecture is:

**DRAM-style periodic semiconductor layouts**

The synthetic dataset and localization evaluation are designed around repeated memory-like structures where periodic ambiguity is a major challenge. The generator models word lines, bit lines, and contact/via patterns at every intersection, with the lattice pitch preserved while critical dimensions vary.

## 4. Dataset Generation

The repository contains a standalone synthetic SEM dataset generator.

Generated pairs consist of:

```text
Reference: 1000 × 1000
Search:    1000 × 1000
```

Each generated sample records the **true center coordinates** of the inserted target pattern as ground truth in `manifest.csv` (`target_center_x`, `target_center_y`).

The generator models controlled semiconductor/SEM variations including:

- geometric scale variation
- rotation variation
- feature-size variation (pitch preserved)
- detector noise, independently drawn per image
- gain / offset (contrast) drift
- gamma response non-linearity
- Gaussian beam blur
- SEM edge brightening
- local jitter and missing / weak features
- Linewidth/CD bias (nm)
- Corner rounding (px)
- Beam astigmatism ratio
- Barrel(+)/pincushion(-) distortion
- Gamma (contrast curve)
- Speckle noise sigma (multiplicative)
- Salt-and-pepper probability








Ground truth is used only for **offline evaluation**. It is never supplied to the localization inference path.

## 5. Repository Structure

```text
DRIFT-SENSE/
├── README.md
├── .gitignore
├── LICENSE
├── dataset_generator.py
├── evaluation_script.py
├── final_production_check.py
├── dataset_visual_generator.py
├── requirements.txt
├── references.md
├── benchmark.csv
├── dataset/              # frozen 30-case benchmark (search / reference / metadata / manifest)
├── test_input/           # demo reference.png + search.png used by the CLI defaults
├── slide_success.png
└── slide_failure.png
```

### File descriptions

| File | Purpose |
|---|---|
| `README.md` | Complete setup and usage instructions |
| `dataset_generator.py` | Generates synthetic reference/search image pairs and records ground truth |
| `evaluation_script.py` | Official localization inference script (`--reference` + `--search` → `RESULT: (x, y)`) |
| `final_production_check.py` | Runs the frozen 30-pair benchmark and reports accuracy / runtime / regressions |
| `dataset_visual_generator.py` | Renders `slide_success.png` and `slide_failure.png` from the benchmark |
| `requirements.txt` | Python dependencies required to reproduce the environment |
| `references.md` | Supporting research and technical references |
| `benchmark.csv` | Frozen baseline results for the 30-case benchmark |
| `dataset/` | Frozen 30-case benchmark (do not regenerate) |
| `test_input/` | Demo images used when no paths are passed to `evaluation_script.py` |
| `slide_success.png` | Successful localization example |
| `slide_failure.png` | Honest failure example |

## 6. Requirements

Recommended environment:

- Python 3.10+
- CPU-based execution
- NumPy
- OpenCV
- Matplotlib (only for `dataset_visual_generator.py`)

No GPU or deep-learning framework is required.

No trained neural-network model is used.

## 7. Installation

Clone the repository:

```bash
git clone https://github.com/VEERU24EC006/DRIFT-SENSE.git
cd DRIFT-SENSE
```

Create a virtual environment.

### Windows PowerShell

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv env
source env/bin/activate
```

Install the exact dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```bash
python --version
python -m pip freeze
```

## 8. Generate a Sample Dataset

The dataset generator creates synthetic image pairs and records their ground-truth locations.

```bash
python dataset_generator.py --pairs 1 --output sample_data
```

Arguments: `--samples N` / `--pairs N` (number of pairs, alias), `--output DIR` (output directory), `--force` (overwrite a non-empty output directory).

This creates a structure similar to:

```text
sample_data/
├── search/
│   └── sample_000_search.png
├── reference/
│   └── sample_000_ref.png
├── metadata/
│   └── sample_000_metadata.json
├── manifest.csv
└── manifest.json
```

`manifest.csv` contains the true target center for every generated pair:

```text
sample_id,search,reference,target_center_x,target_center_y,...
sample_000,search/sample_000_search.png,reference/sample_000_ref.png,452.4,263.2,...
```

Frozen Benchmark Verification

'''text final_production_check.py
'''
provides the final reproducibility check for the locked production implementation.

Run:

'''textpython final_production_check.py
'''

The script runs the locked matcher on the frozen 30-pair benchmark stored in dataset/ and reports:

Mean, median, and worst-case localization error
Pass rate within 0.65 px, 1 px, 2 px, and 5 px
Per-pair and aggregate runtime
Prediction differences against benchmark.csv
Decision differences against benchmark.csv
Detection of new catastrophic regressions

This reproduces the benchmark results reported in Section 15 and verifies that the locked production implementation has not changed its validated behavior.

Submission Visual Generation

'''text
dataset_visual_generator.py 
'''
generates the two visual examples used in the submission.

Run:

'''text
python dataset_visual_generator.py
'''

The script selects:

one success case with localization error ≤ 0.65 px
one honest failure case with error > 5 px

It runs the production matcher and renders:

slide_success.png
slide_failure.png

Each figure contains three panels:

Reference image
Full Search image with the predicted location and true location
Zoomed target region

The failure visualization also reports the diagnosed failure mechanism, such as periodic-lattice lock-on.

### Important

Ground-truth coordinates are generated for **evaluation only**. They are not passed to the inference algorithm.

## 9. Run Localization

The localization entry point is `evaluation_script.py`.

It is run by providing:

1. the path to the reference image
2. the path to the search image

Example:

```bash
python evaluation_script.py --reference sample_data/reference/sample_000_ref.png --search sample_data/search/sample_000_search.png
( or can change name to search.png or reference.png for convenience)

```

Expected output:

```text
RESULT: (452.0000, 263.0000)
```

The returned coordinates are in **Search Image pixel coordinates**.

Running with no arguments uses the demo pair in `test_input/`:

```bash
python evaluation_script.py
```

### Inference requirements

The inference script:

- accepts the image paths as inputs
- does not require ground truth
- does not require source-code modification
- does not require a trained model
- automatically executes the production localization pipeline
- returns the predicted center `(x, y)`

**This is the script intended for external evaluation.**

## 10. Direct Production Matcher

`evaluation_script.py` is the underlying production implementation. Its command-line interface is:

```text
python evaluation_script.py --search SEARCH --reference REFERENCE [--beam_spot_nm BEAM_SPOT_NM] [--output OUTPUT]
```

Optional arguments:

```text
--beam_spot_nm
```

allows beam-spot evaluation when required.

```text
--output
```

writes the resulting evaluation information to a CSV file.

## 11. Localization Pipeline

The production implementation evaluates multiple hypotheses over:

### Geometric scale

```text
0.08
0.09
0.10
0.11
0.12
```

### Rotation

```text
-2°
-1°
0°
+1°
+2°
```

### Feature size

```text
0.50
0.75
1.00
1.25
1.50
```

This forms:

```text
5 × 5 × 5 = 125 hypotheses
```
Controlled Noise and Severity Testing

The dataset was evaluated not only under nominal conditions but also across controlled noise and imaging severity levels. This includes dose/Poisson noise, multiplicative speckle noise, and probabilistic salt-and-pepper corruption, along with charging, beam and raster-related effects.

Severity sweeps were used to determine whether a parameter actually caused localization degradation, rather than optimizing only the correlation score. This prevented unnecessary preprocessing from being added when image quality decreased but localization remained stable.

For each valid hypothesis, DRIFT-SENSE:

1. transforms the reference according to the hypothesis
2. performs normalized cross-correlation
3. extracts multiple strong peaks
4. applies spatial suppression
5. combines and deduplicates candidates
6. retains a bounded candidate pool (48)
7. performs 3×3 Regional Census structural verification on the top candidates
8. selects the final candidate
9. returns its center coordinates

## 12. Why It Handles Periodic Layouts Better

A basic approach would be:

```text
Reference
   ↓
NCC
   ↓
Highest peak
   ↓
(x, y)
```

This can fail in repetitive DRAM structures because multiple sites can have almost identical local appearance.

DRIFT-SENSE instead uses:

```text
Reference + Search
        ↓
Multiple hypotheses
        ↓
Multiple NCC candidates
        ↓
Candidate pool
        ↓
Regional Census verification
        ↓
Final location
```

This allows a structurally incorrect but high-correlation candidate to be compared against other plausible candidates before final selection.

## 13. Numerical Robustness

During development, a pathological behavior in normalized correlation was identified.

When a transformed reference became constant or effectively constant, normalized correlation could produce an artificial perfect response across the search map. These invalid peaks could enter the candidate pool and cause a completely incorrect result.

The final production matcher therefore includes a **template-variance guard** that rejects degenerate hypotheses before normalized correlation.

A second narrowly scoped protection handles pathological near-perfect adaptive NCC responses (returning the raw baseline coordinate instead).

These changes were validated against the frozen benchmark and introduced:

- no prediction regressions
- no decision regressions
- no new catastrophic regressions

Observed Pathologies and Targeted Fixes

Two important NCC pathology mechanisms were identified through forensic testing.

Zero-variance template pathology

Some feature-size hypotheses became completely constant. OpenCV TM_CCOEFF_NORMED could then produce an artificial 1.0 response across the entire search image, creating arbitrary false peaks and poisoning the candidate pool.

The production matcher now performs a template-variance check before NCC and skips degenerate hypotheses.

Near-constant template / salt-and-pepper pathology

A separate failure was observed with severe salt-and-pepper corruption. Some transformed templates became numerically near-constant rather than exactly zero-variance, producing pathological near-perfect NCC responses.

The production build therefore contains a narrow safety gate for pathological near-perfect Adaptive NCC responses, preventing such a response from incorrectly overriding a valid RAW localization.

These protections were validated against the frozen 30-case benchmark with:

0 prediction regressions
0 decision regressions
0 new catastrophic regressions

External Noise Validation

Additional external testing was performed on independently generated speckle cases and original pathological examples.

The analysis distinguished two different failure mechanisms:

Speckle/noise-related NCC pathology, which was mitigated by the final numerical safeguards.
True periodic-layout ambiguity, where a valid but incorrect repeated structure can still outrank the correct candidate.

Therefore, noise-pathology failures should not be confused with the remaining periodic-layout limitation.

## 14. Evaluation Method

For offline evaluation, the predicted center is compared with the generated ground-truth center.

Euclidean localization error:

```text
error = sqrt((pred_x - GT_x)^2 + (pred_y - GT_y)^2)
```

The benchmark reports:

- mean localization error
- median localization error
- worst-case error
- pass rate within selected tolerances
- runtime per image pair

Ground truth is never used during inference.

## 15. Final Benchmark

The final locked production evaluator was tested on **30 image pairs**.

| Metric | Result |
|---|---|
| Test pairs | 30 |
| Within 0.65 px | 93.3% |
| Within 1 px | 93.3% |
| Within 2 px | 93.3% |
| Within 5 px | 93.3% |
| Median error | 0.403 px |
| Mean error | 27.687 px |
| Worst-case error | 497.848 px |
| Mean runtime | ~9.9 s / pair |
| Median runtime | ~8.3 s / pair |
| Worst runtime | ~16.1 s / pair |

The mean error is dominated by a small number of catastrophic cases; the median error and pass rates better represent typical localization behavior. Reproduce these numbers with `final_production_check.py`.

## 16. Known Failure Modes

DRIFT-SENSE is not presented as a universally perfect solution.

A known failure mechanism is **periodic candidate ambiguity**.

`sample_010` produced approximately:

```text
Ground truth: (684.67, 572.44)
Prediction:   (907.00, 127.00)
Error:        497.85 px
```

The true candidate existed close to the correct location, but another repeated structure received a stronger score during candidate selection.

This limitation is intentionally documented rather than hidden.

Additional external stress testing showed sensitivity to severe:

- radial distortion
- row jitter
- geometric distortion
- highly repetitive ambiguity

These are known research limitations of the current locked build.

Known Failure Modes Clarification

The current build has already mitigated the identified zero-variance and pathological near-perfect NCC failures associated with extreme noise cases.

The remaining known limitations are primarily:

Severe barrel/pincushion distortion
Severe row jitter
Strong geometric deformation
Periodic candidate ambiguity where a false repeated structure receives a stronger overall score

These remain honest limitations of the locked implementation and were not addressed with unvalidated last-minute algorithmic changes.

## 17. Technology Stack

The software prototype uses:

- Python
- NumPy
- OpenCV
- Matplotlib (visualization only)
- standard Python utilities

The final system is classical computer vision and does not use a deep-learning model.

## 18. Hardware Implementation Direction

The current repository contains the validated **software reference implementation**.

The intended future acceleration path is:

```text
Software Algorithm
       ↓
Dataflow Decomposition
       ↓
DMA / AXI4-Stream
       ↓
BRAM Line Buffers
       ↓
DSP / MAC Acceleration
       ↓
srv32 RISC-V Control
       ↓
CLIC Interrupt Handling
```

This hardware direction is separate from the Python runtime and is intended as a future implementation/acceleration path after algorithmic validation.

## 19. References

Technical and scientific references used for the project are listed in:

```text
references.md
```

These include references covering:

- normalized cross-correlation
- Census transform
- image registration
- SEM imaging effects
- semiconductor inspection considerations
- noise and acquisition effects

## 20. Reproducibility

For reproducible experiments:

1. Use the Python environment described by `requirements.txt`.
2. Generate a fresh synthetic pair using `dataset_generator.py`.
3. Run `evaluation_script.py` using only the reference and search image paths.
4. Record the predicted `(x, y)`.
5. Compare against ground truth only during offline evaluation.

Do not pass ground-truth coordinates to the inference script.

## 21. Project Status

**DRIFT-SENSE submission build: FINAL / LOCKED**

The final system:

- targets DRAM-style periodic layouts
- handles scale, rotation and feature-size variation
- retains multiple localization candidates
- uses Regional Census structural verification
- includes protection against degenerate normalized-correlation cases
- was regression-tested on the frozen benchmark
- provides a standalone inference path
- provides a synthetic dataset generator
- provides a reproducible software implementation
- provides a future path toward hardware acceleration

For external scoring, **use `evaluation_script.py` as the official localization entry point**.

## Quick Start

After installation:

```bash
python dataset_generator.py --pairs 1 --output sample_data
```

Then:

```bash
python evaluation_script.py --reference sample_data/reference/sample_000_ref.png --search sample_data/search/sample_000_search.png
```

Expected result:

```text
RESULT: (x, y)
```

For external scoring, run:

python evaluation_script.py --reference <reference_image> --search <search_image>

The script performs inference without ground truth and prints:

RESULT: (x, y)

For benchmark validation, use:

python final_production_check.py

For submission visualization, use:

python dataset_visual_generator.py


No source-code modification or ground-truth input is required.
