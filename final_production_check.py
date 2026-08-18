import os, csv, time
import numpy as np
import evaluation_script as ev

BENCH = "benchmark.csv"
XY_TOL = 5.1e-5
KNOWN_FAILURES = {"sample_010", "sample_025"}


def load_frozen():
    rows = {}
    with open(BENCH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = os.path.basename(r["search"]).replace("_search.png", "")
            rows[stem] = r
    return rows


def main():
    frozen = load_frozen()
    manifest = {r["sample"]: r for r in ev.load_manifest()}
    samples = [s for s in frozen if s in manifest]

    errs, rts = [], []
    pred_diffs, dec_diffs, new_cat = [], [], []
    for s in samples:
        m = manifest[s]
        search = ev.load_gray(ev.dataset_path(m["search"]))
        ref = ev.load_gray(ev.dataset_path(m["reference"]))
        gt = (m["gt_x"], m["gt_y"])
        t0 = time.perf_counter()
        r = ev.match_pair(search, ref)
        rt = (time.perf_counter() - t0) * 1000.0
        e = float(np.hypot(r["x"] - gt[0], r["y"] - gt[1]))
        errs.append(e); rts.append(rt)

        f = frozen[s]
        old_e = float(f["error_px"]) if f["error_px"] else np.nan
        fp = (float(f["pred_x"]), float(f["pred_y"]))
        if abs(r["x"] - fp[0]) > XY_TOL or abs(r["y"] - fp[1]) > XY_TOL:
            pred_diffs.append(s)
        if r["decision"] != f["decision"].strip():
            dec_diffs.append(s)
        if e > 5.0 and old_e <= 5.0 and s not in KNOWN_FAILURES:
            new_cat.append((s, old_e, e))

    a, rr = np.array(errs), np.array(rts)
    print("=" * 72)
    print("FINAL PRODUCTION CHECK | current evaluation_script.py | frozen 30-sample")
    print("=" * 72)
    print(f"sample count           : {len(samples)}")
    print(f"runtime  mean/median/worst : {rr.mean():7.1f} / {np.median(rr):7.1f} / {rr.max():7.1f} ms")
    print(f"error    mean/median/worst : {a.mean():7.3f} / {np.median(a):7.3f} / {a.max():7.3f} px")
    print(f"<=0.65 px : {(a <= 0.65).mean() * 100:5.1f}%   <=1 px : {(a <= 1.0).mean() * 100:5.1f}%   "
          f"<=2 px : {(a <= 2.0).mean() * 100:5.1f}%   <=5 px : {(a <= 5.0).mean() * 100:5.1f}%")
    print(f"prediction diffs vs frozen : {len(pred_diffs)} {pred_diffs if pred_diffs else ''}")
    print(f"decision diffs vs frozen   : {len(dec_diffs)} {dec_diffs if dec_diffs else ''}")
    print(f"new catastrophic regressions (new>5 & old<=5, excl. sample_010/025): {len(new_cat)}")
    for s, old_e, new_e in new_cat:
        print(f"    {s}: old={old_e:.3f} -> new={new_e:.3f} px")
    print("=" * 72)


if __name__ == "__main__":
    main()