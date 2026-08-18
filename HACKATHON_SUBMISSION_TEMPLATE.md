# Hackathon Idea Submission Template — Drift-Sense

Fill the bracketed `[...]` placeholders (team names, contacts, links) before
submitting. Every other cell is ready to drop into the PPT.

---

## Slide 1 — Team Details

| Field | Value |
|---|---|
| **Team name** | [TEAM NAME] |
| **Members (name — role)** | [Member 1] — Algorithm & Dataset Design; [Member 2] — Validation & Testing; [Member 3] — Deployment & Docs |
| **College / organization** | [COLLEGE NAME] |
| **Contact** | [email]; [phone / LinkedIn]; [country/city] |

---

## Slide 2 — Problem Statement Addressed

**Selected track:** *Drift-Sense: Navigation-Error Recovery*

**Why it matters.** In semiconductor wafer inspection, a reference image of a
die/feature pattern is captured once (design or golden template), then the
inspection tool must re-locate that same pattern on a newly acquired image of a
new die. The acquired image is never identical to the reference: the stage may
have drifted (navigation error), the tool may acquire at a different
magnification (scale), the wafer may be rotated, and detector gain/contrast may
have drifted. Failure to recover the correct location means the tool inspects
the wrong structure — producing false defects, missing real ones, and wasted
throughput. The problem is hardest for **periodic layouts** (repeating word
lines, contacts, fins), where every lattice period looks like a valid match and
a naive template matcher can latch onto the wrong instance with a perfect score.

---

## Slide 3 — Idea Description

- **Architecture style:** We generate and evaluate **DRAM-style**
  layouts (word-line / bit-line / contact lattice — periodic horizontal
  word-lines and vertical bit-lines crossing at right angles with a
  contact/via dot at every intersection).
- **Localization algorithm:** **Classical (non-deep-learning)**: a 125-hypothesis
  grid (5 feature-size × 5 scale × 5 rotation transforms of the reference),
  parallel normalized cross-correlation (NCC), followed by a regional Census
  transform validation stage, plus explicit pathological-NCC and zero-variance
  safety gates.
- **Why better than simple template matching:** simple single-template NCC fails
  on (a) the **10x scale difference** (a fixed-size template cannot cover it),
  (b) **rotation** and feature-size variation, and (c) **periodic ambiguity**,
  where NCC returns 1.0 for every lattice instance. We handle scale/rotation/
  feature-size by explicitly searching that parameter space, and periodic
  ambiguity by a **regional ordering-based (Census) validator** that rejects
  periodic impostors, and pathological near-perfect matches by a safety gate.

---

## Slide 4 — Proposed Solution

**Pipeline (input pair → output `(x, y)`):**

```
 reference.png ──┬─► [feature-size morph (erode/dilate) ×5]
                 │   [resample ×5 scales]  [rotate ±2° ×5]
                 │            │
                 │   ┌────────▼────────┐
                 │   │ 125 templates   │   search.png ──► [beam blur (evaluator)]
                 │   └────────┬────────┘          │
                 │            ▼                   ▼
                 │   parallel cv2.matchTemplate (TM_CCOEFF_NORMED, 4 threads)
                 │            ▼
                 │   top-8 peaks/hypothesis ─► spatial dedup ─► 48-candidate pool
                 │            ▼
                 │   regional 3×3 Census re-score of top-20
                 │            ▼
                 │   selection (ZNCC / CENSUS_GATE / CENSUS_SECOND_CHANCE)
                 │            ▼
                 └──► adaptive safety gate (ZNCC≥0.999999 & RAW lower → RAW)
                              ▼
                       RESULT: (x, y)
```

**Dataset generator design.** Synthetic parent layout (11000×11000 px) with
pitch-preserved feature scaling; crops a 1000×1000 reference at the exact
target and a 1000×1000 search at a random window; degrades **reference and
search independently**. Noise/augmentation (all cited — see Slide 9 /
`references.md`): Gaussian beam blur, Poissonian-Gaussian detector noise
 (search noisier than reference), SEM edge-brightening (unsharp mask),
 gain/offset drift, gamma non-linearity, 10x `INTER_AREA` resampling, ±2°
 rotation, feature-size scaling ×{0.5,1.0,2.0}, local jitter, 2% missing/weak
 features. **Ground truth** is the parent-geometry center written to
 `manifest.csv` — deterministic, not re-measured.

**Key design decisions.**
- 10x scale handled by a 5-scale NCC grid (0.08–0.12) rather than a single
  naive template.
- Periodic ambiguity resolved at a **non-linear, monotonic descriptor level**
  (Census), which is robust to gain/impulse noise and cannot be fooled by the
  brightness-invariance of NCC.
- Zero-variance templates skipped (OpenCV `TM_CCOEFF_NORMED` returns all-ones
  for `templNorm<eps`); near-perfect Adaptive matches that beat RAW are treated
  as pathological and the RAW baseline is returned.

---

## Slide 5 — Innovation & Uniqueness

1. **More realistic generator than baseline.** Reference and search are
   *independently* degraded, with pitch-preserved feature scaling, local jitter,
   and missing/weak features — matching real CD-SEM acquisition, not a clean
   paste-and-shift.
2. **Periodic-ambiguity handling beyond NCC.** A regional Census validator
   separates the true lattice instance from identical-looking periodic
   impostors that score ZNCC ≈ 1.0.
3. **Novel 10x-scale treatment.** Instead of one resized template, we search an
   explicit scale × feature-size × rotation grid with parallel NCC and
   morphology-based feature scaling — exact where downscaling loses contrast.
4. **Pathological-match safety.** Two documented failure modes found in testing
   (zero-variance templates; impulse-noise-induced 1.0 responses) are handled by
   explicit guards, so the matcher degrades gracefully instead of emitting a
   confident wrong coordinate.
5. **Fully reproducible, CPU-only, no deep learning** — transparent, debuggable,
   and fast to certify.

---

## Slide 6 — Results

Frozen 30-case benchmark (1000×1000 pairs, consumer CPU):

| Metric | Value |
|---|---|
| Predictions ≤ 0.65 px of true center | **93.3%** (28/30) |
| Mean / median / worst error | 27.7 px / 0.4 px / 497.8 px |
| Runtime per 1000×1000 pair | ≈ 3.6 s (4 OpenCV + 4 worker threads) |

**Success case (visual):** `dataset/sample_000` — reference, search, predicted
`(452.4, 263.2)` vs true `(452.4, 263.2)`, error 0.0 px.

**Honest failure case (visual):** `dataset/sample_010` and `sample_025` are
highly-repetitive layouts; the matcher locks onto a *correct-but-different*
lattice instance that is visually indistinguishable at this resolution (a
global, not local, ambiguity). Error 497.8 px is a single lattice period. We
report these honestly instead of masking them, and they motivate the Census
validation stage.

---

## Slide 7 — Technology & Feasibility

| Item | Detail |
|---|---|
| Language / stack | Python 3.10+; **NumPy**, **OpenCV** (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`, morphology, warp) |
| Runtime packages | `numpy`, `opencv-python` only |
| Hardware | CPU-only (Windows 11 / x86-64, consumer desktop) |
| Dataset generation time | [e.g. ~1–2 min per 30 pairs] |
| Inference time per pair | ≈ 3.6 s (1000×1000) |
| Model type / size | N/A — classical algorithm, no weights |

---

## Slide 8 — GitHub & Video Link

- **GitHub repository (public, mandatory):** https://github.com/[owner]/[repo]
  — contains README (setup), `dataset_generator.py` (DRAM-style, GT),
  `evaluation_script.py` (the inference script AM runs),
  `final_production_check.py`, `requirements.txt`,
  `references.md`, frozen benchmark.
- **Video (optional but recommended):** [YouTube/Vimeo link] — algorithm running
  on a sample pair (generate → inference → RESULT line).

---

## Slide 9 — References

All citations are listed with a mapping to each augmentation/noise choice in
`references.md`; the key sources:

1. Goldstein et al., *Scanning Electron Microscopy and X-ray Microanalysis*, Springer, 2017.
2. Postek & Vladár, *Does your SEM really tell the truth?*, Proc. SPIE 9236, 2014.
3. Sutton et al., *SEM for quantitative small/large deformation measurements*, Exp. Mechanics 47, 2007.
4. Foi et al., *Practical Poissonian-Gaussian Noise Modeling for Raw Data*, IEEE TIP 17(10), 2008.
5. Lewis, *Fast Normalized Cross-Correlation*, ILM, 1995.
6. Briechle & Hanebeck, *Template Matching Using Fast Normalized Cross Correlation*, Proc. SPIE 4387, 2001.
7. OpenCV docs/source, `matchTemplate` / `templmatch.cpp`.
8. Zabih & Woodfill, *Non-parametric local transforms for computing visual correspondence*, ECCV, 1994.
9. Zitová & Flusser, *Image registration methods: a survey*, IVC 21(11), 2003.
10. Gonzalez & Woods, *Digital Image Processing*, 4th ed., Pearson, 2018.
11. Mack, *Fundamental Principles of Optical Lithography*, Wiley, 2007.
