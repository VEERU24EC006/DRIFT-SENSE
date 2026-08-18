# DRIFT-SENSE
DRIFT SENSE AI-Powered Navigation-Error Recovery for Wafer Inspection Tools. SUBMISSION FOR SEMICON 2026.
DRIFT-SENSE
AI-Powered Navigation-Error Recovery for Semiconductor Wafer Inspection

DRIFT-SENSE is a classical computer-vision localization system designed for DRAM-style periodic SEM layouts. Given a small reference image and a large search image, the system identifies where the reference pattern appears in the search image and returns the predicted center coordinates (x, y).

The system addresses the Navigation-Error Recovery problem: an inspection tool may revisit a wafer site with a positioning error caused by stage drift, thermal effects, vibration, mechanical tolerances, or imaging variation. Because semiconductor layouts are highly repetitive, the wrong site can look almost identical to the intended site. DRIFT-SENSE therefore does not rely on a single template-matching peak.

1. Problem Definition
Inputs
Reference image: approximately 100 × 100 pixels
Search image: 1000 × 1000 pixels
The reference represents the target structure at approximately 10× higher magnification than the search image.
Expected geometric scale relationship is approximately 9×, 10×, or 11×.
Small rotations and feature/morphology variations may be present.

The inference system returns:

(x, y)

where (x, y) is the predicted center of the reference pattern in Search Image pixel coordinates.

Ground truth is used only for offline evaluation. The production inference path never receives ground-truth coordinates. 
Key Approach

DRIFT-SENSE is a classical computer-vision system, not a deep-learning model.

The production pipeline is:

Reference + Search Images
          ↓
Multi-hypothesis generation
(scale × rotation × feature size)
          ↓
Normalized Cross-Correlation (NCC)
          ↓
Multiple peak extraction
          ↓
Spatial suppression + candidate deduplication
          ↓
3 × 3 Regional Census verification
          ↓
Final candidate selection
          ↓
Predicted center (x, y)


Why multiple candidates?

Simple template matching often assumes that the highest correlation peak is the correct location. This is unreliable for periodic semiconductor layouts because many regions can contain nearly identical structures.

DRIFT-SENSE retains multiple promising candidates and applies structural verification before selecting the final location.

Why Regional Census?

NCC measures pixel-level similarity and can be ambiguous when repeated structures look similar. Regional Census verification adds local structural information and helps distinguish competing periodic candidates without relying entirely on absolute intensity.


Dataset Generator

The repository contains a standalone synthetic SEM dataset generator.

The generator is designed to produce:

Reference images
Search images
Ground-truth center coordinates
Architecture-dependent synthetic patterns
Controlled imaging and noise variations

The primary validation and presentation focus of DRIFT-SENSE is DRAM-style periodic layouts.

Controlled variations

The synthetic dataset can include:

Geometric scale variation
Rotation variation
Feature-size variation
Dose / Poisson noise
Multiplicative speckle noise
Salt-and-pepper corruption controlled by probability
Charging effects
Beam-related effects
Pattern-collapse variation
CD / line-width variation
Corner-rounding variation
Raster / scan-related distortions

These controls are intended to model realistic SEM acquisition and semiconductor-pattern variability rather than arbitrary image augmentation.

Ground truth

For every generated pair, the generator records the true center of the inserted target pattern.

Ground truth is used for:

Localization-error calculation
Regression/pass-rate analysis
Failure analysis
Benchmark validation

. Production Localization Details

The locked production matcher evaluates:

Geometric scale:
0.08, 0.09, 0.10, 0.11, 0.12


Rotation:
-2°, -1°, 0°, +1°, +2°


Feature-size factor:
0.50, 0.75, 1.00, 1.25, 1.50

This produces:

5 × 5 × 5 = 125 hypotheses

For each hypothesis:

Generate the corresponding transformed reference.
Reject numerically degenerate constant templates before NCC.
Perform normalized template matching.
Extract multiple high-quality peaks.
Apply spatial suppression.
Combine and deduplicate candidates.
Retain a bounded candidate pool.
Evaluate local structural similarity using Regional Census.
Select the final validated candidate.
Convert the match location into the target center (x, y).
11. Numerical Robustness

During forensic validation, we identified a failure mode involving normalized template matching.

A constant or effectively constant template has undefined normalized correlation. In OpenCV, such a case can produce an artificial response of 1.0 across the response map, creating fake peaks that can poison candidate selection.

DRIFT-SENSE therefore checks template variance before invoking normalized correlation and skips degenerate hypotheses.

A second narrow safeguard prevents a pathological near-perfect adaptive NCC result from incorrectly overriding a valid RAW result when the response is numerically suspicious.

Both protections were regression-tested against the frozen benchmark.

12. Benchmark Results

The final locked production evaluator was tested on a frozen 30-pair benchmark.

Metric	Result
Test pairs	30
Accuracy within 0.65 px	93.3%
Accuracy within 1 px	93.3%
Accuracy within 2 px	93.3%
Accuracy within 5 px	93.3%
Median error	0.403 px
Mean error	27.687 px
Worst error	497.848 px
Mean runtime	~4.02 s / pair
Median runtime	~3.73 s / pair
Worst runtime	~6.51 s / pair

The mean error is strongly affected by a small number of catastrophic failures; the median error and pass rates better represent normal localization behavior.

Final production hardening resulted in:

0 prediction regressions on the frozen benchmark
0 decision regressions
0 new catastrophic regressions
13. Honest Failure Behavior

DRIFT-SENSE is not claimed to be perfect.

One known failure mode is periodic ambiguity, where a repeated semiconductor structure can produce a false candidate whose correlation score is slightly stronger than the true location.

Example frozen case:

Ground truth: (778.73, 623.45)
Prediction:   (779.00, 915.00)
Error:        291.55 px

The true candidate existed and was close to the correct location, but another repetitive structure outranked it during candidate selection.

This limitation is documented intentionally.

Additional external stress testing showed sensitivity to:

severe barrel/pincushion distortion
severe row jitter
extreme repetitive ambiguity
pathological acquisition conditions outside the validated distribution

These remain known limitations rather than claimed solved cases.

14. Why This Approach?
Simple template matching
Reference
   ↓
Correlation map
   ↓
Global maximum
   ↓
(x, y)
DRIFT-SENSE
Reference + Search
        ↓
Scale / rotation / feature hypotheses
        ↓
Multiple NCC candidates
        ↓
Candidate pool
        ↓
Regional Census verification
        ↓
Validated (x, y)

The difference matters for periodic DRAM layouts because the strongest pixel correlation is not necessarily the intended site.

15. Hardware Acceleration Direction

The submitted system is a validated software prototype.

The intended future hardware implementation maps computationally intensive dataflow into:

DMA for image movement
AXI4-Stream for streaming data paths
BRAM line buffers for local image storage
DSP/MAC structures for correlation-heavy operations
srv32 RISC-V control
CLIC interrupt handling

The Python implementation serves as the algorithmic reference against which a future hardware accelerator can be verified.

The hardware architecture is not required to run this Python repository.

16. Evaluation Methodology

For each image pair:

Load reference and search images.
Run inference without ground truth.
Record predicted (x,y).
Compare prediction with ground-truth metadata offline.
Compute Euclidean localization error.
Aggregate:
Mean error
Median error
Worst-case error
Pass rate at selected tolerances
Runtime per image pair

Ground truth is never used during inference.

17. References

Supporting literature and technical references are listed in:

REFERENCES.md

Theoretical foundations include:

Normalized cross-correlation and template matching
Census transform / non-parametric local transforms
Computer-vision image registration and matching
SEM imaging and acquisition effects
18. Limitations

The current production system should not be interpreted as universally robust to every possible SEM artifact.

Known limitations include:

Severe radial distortion
Severe row jitter
Extreme repetitive false matches
Pathological acquisition conditions outside the validated dataset distribution

The project prioritizes preserving validated behavior and avoiding regressions over adding unvalidated corrections.

19. Reproducibility and Safety

Before changing the production matcher:

Preserve a backup.
Freeze the benchmark.
Change one factor at a time.
Evaluate localization error, not only correlation score.
Run the complete frozen regression.
Never use ground truth inside inference.
Never replace the locked production implementation with an unvalidated experiment.

Diagnostic and research scripts are intentionally excluded from the minimal public submission package.

20. Project Status

DRIFT-SENSE submission build: FINAL / LOCKED

The final validated system:

targets DRAM-style periodic layouts
handles scale, rotation, and feature-size variation
evaluates multiple candidates rather than trusting one correlation peak
uses Regional Census structural verification
contains verified NCC-degeneracy protections
has been regression-tested on the frozen benchmark
provides a reproducible software inference path
provides a clear path toward future hardware acceleration
