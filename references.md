# References — Justification of Noise, Augmentation & Matching Choices

This document lists the sources that justify every degradation, augmentation,
and matching-method decision in the Drift-Sense dataset generator and
localization algorithm. Each item maps to the relevant choice(s). The same
references appear in the competition slide deck (Slide 9).

---

## A. SEM / electron-microscope imaging & degradation realism

1. **J. I. Goldstein, D. E. Newbury, J. R. Michael, N. W. M. Ritchie, J. H. J. Scott, D. C. Joy** — *Scanning Electron Microscopy and X-ray Microanalysis*, 4th ed., Springer, 2017.
   Justifies: electron-beam blur modeled as a Gaussian point-spread function;
   detector noise; contrast/gain and brightness (offset) drift; intensity
   response non-linearity (gamma). The generator's blur, gain/offset and gamma
   stages follow SEM image-formation practice from this reference.

2. **M. T. Postek, A. E. Vladár** — *Does your SEM really tell the truth? — How to properly use imaging and metrology in the SEM*, Proc. SPIE **9236**, 2014.
   Justifies: the requirement that a localization method remain accurate under
   realistic SEM artifacts (noise, drift, vibration) rather than only on clean
   synthetic images — the motivation for testing on degraded, independently
   degraded reference/search pairs.

3. **M. A. Sutton, N. Li, D. C. Joy, A. P. Reynolds, X. Li** — *Scanning electron microscopy for quantitative small and large deformation measurements — Part I*, Experimental Mechanics **47**, 2007.
   Justifies: modeling image-to-image distortion/drift between acquisitions and
   why template-rigidity assumptions fail — the rationale for a multi-scale,
   multi-rotation hypothesis grid and a robust (non-linear) Census similarity
   stage.

## B. Detector / noise models

4. **A. Foi, M. Trimeche, V. Katkovnik, K. Egiazarian** — *Practical Poissonian-Gaussian Noise Modeling and Fitting for Single-Image Raw-Data*, IEEE Trans. Image Processing **17**(10), 2008.
   Justifies: the detector-noise model. Electron counting is Poisson; at the
   synthetic intensities used here the Poisson statistics are well approximated
   by a signal-dependent Gaussian `N(0, σ)` with σ in 2–5 DN, matching the
   standard approximation used for SEM detectors.

## C. Localization / template-matching methodology

5. **J. P. Lewis** — *Fast Normalized Cross-Correlation*, Industrial Light & Magic, 1995.
   Justifies: using normalized cross-correlation (NCC) as the primary
   similarity measure — brightness/gain-invariant, and the basis of
   `cv2.matchTemplate(TM_CCOEFF_NORMED)` used in `evaluation_script.py`.

6. **K. Briechle, U. D. Hanebeck** — *Template Matching Using Fast Normalized Cross Correlation*, Proc. SPIE **4387**, 2001.
   Justifies: the mathematical treatment of NCC and its normalization; defines
   the well-posedness (non-zero template energy) requirement behind the
   zero-variance guard.

7. **OpenCV documentation & source** — `cv2.matchTemplate` / `cv2.TM_CCOEFF_NORMED`, `modules/imgproc/src/templmatch.cpp` (`common_matchTemplate`).
   Justifies: the observed all-ones response for zero-variance templates
   (`templNorm < DBL_EPSILON`) and the guard added in `generate_candidates` to
   skip mathematically undefined NCC hypotheses.

8. **R. Zabih, J. Woodfill** — *Non-parametric local transforms for computing visual correspondence*, Proc. ECCV, 1994.
   Justifies: the Census transform used as the validation stage — a monotonic
   (ordering-based) descriptor robust to gain, offset, and impulse outliers,
   which disambiguates the periodic-lattice false matches that pure NCC cannot.

9. **B. Zitová, J. Flusser** — *Image registration methods: a survey*, Image and Vision Computing **21**(11), 2003.
   Justifies: the overall feature- + area-based registration pipeline (feature
   transform of the reference, area-based similarity scoring, refinement via
   regional validation) and the handling of scale/rotation parameter search.

## D. Image processing / resampling / geometry

10. **R. C. Gonzalez, R. E. Woods** — *Digital Image Processing*, 4th ed., Pearson, 2018.
    Justifies: `INTER_AREA` resampling for the 10x scale reduction (area-average
    anti-aliasing), morphological erosion/dilation for feature-size variation,
    Gaussian filtering, and gamma correction.

## E. Semiconductor layout / metrology context

11. **C. A. Mack** — *Fundamental Principles of Optical Lithography: The Science of Microfabrication*, Wiley, 2007.
    Justifies: the pitch-preserved periodic-lattice layout model (fixed lattice,
    scaled critical dimensions) used for the DRAM-style (word lines / bit
    lines / contacts) synthetic parent, mirroring real periodic device arrays.

---

## Mapping table (generator stages -> citations)

| Degradation / choice | Generator stage | Citation(s) |
|---|---|---|
| Gaussian beam blur (PSF) | `degrade()` blur | [1] |
| Detector (Poissonian-Gaussian) noise | `degrade()` `N(0,2..4)` ref / `N(0,3..6)` search | [4] |
| Search noisier than reference | larger detector-noise σ on search | [4], test-data rule |
| Gain / offset (contrast) drift | `x*N(0.92,1.08)+N(-6,6)` | [1], [2] |
| SEM edge brightening | unsharp-mask edge enhancement | [1], [10] |
| Gamma / response non-linearity | `(x/255)^gamma*255` | [1], [10] |
| 10x scale reduction | `cv2.resize` `INTER_AREA` | [10] |
| Rotation ±2° | `cv2.warpAffine` | [9] |
| Feature-size variation (pitch preserved) | `generate_parent` | [11] |
| Local jitter / missing / weak features | per-cell jitter, 2% dropouts | [11] |
| Independent reference/search degradation | separate `rng` draws per image | [2] |
| NCC primary similarity | `TM_CCOEFF_NORMED` | [5], [6] |
| Zero-variance guard | `generate_candidates` | [6], [7] |
| Census validation stage | `regional_census` | [8] |
| Multi-scale/rotation hypothesis grid | 125-hypothesis search | [3], [9] |
