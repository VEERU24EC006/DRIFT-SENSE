import os,csv,time,argparse,cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor,as_completed

def ensure_opencv_threads(target_threads=4):
    if cv2.getNumThreads() != target_threads:
        cv2.setNumThreads(target_threads)

ensure_opencv_threads()

DATASET="dataset"
MANIFEST=os.path.join(DATASET,"manifest.csv")
MANIFEST_JSON=os.path.join(DATASET,"manifest.json")
TEST_INPUT="test_input"
OUT_DIR=os.path.join(DATASET,"baseline_results")
OUT_CSV=os.path.join(OUT_DIR,"inference_results.csv")

SCALES=(0.08,0.09,0.10,0.11,0.12)
ANGLES=(-2.0,-1.0,0.0,1.0,2.0)
FEATURE_FACTORS=(0.50,0.75,1.00,1.25,1.50)
FEATURE_BASE_WIDTH_PX=62.0
PEAKS_PER_HYPOTHESIS=8
CANDIDATE_POOL=48
CENSUS_EVAL_LIMIT=20
ZNCC_AMBIGUITY=0.035
ZNCC_SECOND_CHANCE=0.065
REGIONAL_MEAN_GAIN=0.015
REGIONAL_MIN_GAIN=0.008
SECOND_MEAN_GAIN=0.025
SECOND_MIN_GAIN=0.012
SPATIAL_DEDUP_PX=8.0
CENSUS_RADIUS=2
CENSUS_GRID=3

PARALLEL_WORKERS=4
PARALLEL_OPENCV_THREADS=1

def load_gray(path):
    img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if img is None: raise RuntimeError(f"Cannot read: {path}")
    if img.ndim==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img,dtype=np.float32)

def load_raw(path):
    img=cv2.imread(path,cv2.IMREAD_UNCHANGED)
    if img is None: raise RuntimeError(f"Cannot read: {path}")
    if img.ndim==3: img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return np.ascontiguousarray(img)

def image_signature(path):
    img=load_raw(path)
    return (img.shape,img.dtype.str,hash(img.tobytes()))

def apply_beam_spot(search,spot_nm):
    if spot_nm<=0:return search
    sigma=float(spot_nm)/(2.355*10.0)
    if sigma<0.01:return search
    return np.ascontiguousarray(cv2.GaussianBlur(search,(0,0),sigmaX=sigma,sigmaY=sigma),dtype=np.float32)

def zncc_baseline(search,ref):
    template=cv2.resize(ref,(max(8,round(ref.shape[1]*0.1)),max(8,round(ref.shape[0]*0.1))),interpolation=cv2.INTER_AREA)
    response=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
    _,score,_,loc=cv2.minMaxLoc(response)
    h,w=template.shape
    return {"x":loc[0]+w/2.0,"y":loc[1]+h/2.0,"score":float(score)}

def transform_reference(ref,scale,angle,resize_cache=None,cache_key=None):
    h,w=ref.shape
    if resize_cache is not None and cache_key is not None:
        out=resize_cache.get(cache_key)
        if out is None:
            out=cv2.resize(ref,(max(8,round(w*scale)),max(8,round(h*scale))),interpolation=cv2.INTER_AREA)
            resize_cache[cache_key]=out
    else:
        out=cv2.resize(ref,(max(8,round(w*scale)),max(8,round(h*scale))),interpolation=cv2.INTER_AREA)
    if abs(angle)<1e-9:return np.ascontiguousarray(out,dtype=np.float32)
    h,w=out.shape
    M=cv2.getRotationMatrix2D((w/2.0,h/2.0),angle,1.0)
    return np.ascontiguousarray(cv2.warpAffine(out,M,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT),dtype=np.float32)

def build_morph_base(ref,feature_factor):
    """Feature-specific morphed base; depends only on feature_factor."""
    out=np.ascontiguousarray(ref,dtype=np.float32)
    radius=int(round(abs(float(feature_factor)-1.0)*FEATURE_BASE_WIDTH_PX/2.0))
    if radius>=1:
        k=2*radius+1
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
        out=cv2.erode(out,kernel) if feature_factor<1.0 else cv2.dilate(out,kernel)
    return out

def transform_reference_feature(ref,feature_factor,scale,angle,morph_cache=None,resize_cache=None):
    """Build a transformed template from the (possibly cached) feature base.

    morph_cache maps feature_factor -> morphed base (computed once per
    match_pair call); resize_cache maps (feature_factor, scale) -> resized base
    shared across the 5 rotation hypotheses of the same scale.
    """
    if morph_cache is not None:
        base=morph_cache.get(feature_factor)
        if base is None:
            base=build_morph_base(ref,feature_factor)
            morph_cache[feature_factor]=base
    else:
        base=build_morph_base(ref,feature_factor)
    if resize_cache is not None:
        return transform_reference(base,scale,angle,resize_cache=resize_cache,cache_key=(feature_factor,scale))
    return transform_reference(base,scale,angle)

def top_peaks(response,k,min_distance=10):
    work=response.copy()
    peaks=[]
    for _ in range(k):
        _,score,_,loc=cv2.minMaxLoc(work)
        if not np.isfinite(score):break
        x,y=loc
        peaks.append((float(score),int(x),int(y)))
        y0=max(0,y-min_distance);y1=min(work.shape[0],y+min_distance+1)
        x0=max(0,x-min_distance);x1=min(work.shape[1],x+min_distance+1)
        work[y0:y1,x0:x1]=-np.inf
    return peaks

CENSUS_LUT=np.array([bin(i).count("1") for i in range(256)],dtype=np.uint8)

def census_bits(img,radius=CENSUS_RADIUS):
    h,w=img.shape
    if h<=2*radius or w<=2*radius:return None
    center=img[radius:h-radius,radius:w-radius]
    bits=np.zeros(center.shape,dtype=np.uint32)
    bit=0
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dx==0 and dy==0:continue
            neighbor=img[radius+dy:h-radius+dy,radius+dx:w-radius+dx]
            bits|=((neighbor>=center).astype(np.uint32)<<bit)
            bit+=1
    return bits

def census_similarity(a,b):
    if a.shape!=b.shape:return np.nan
    ca=census_bits(a);cb=census_bits(b)
    if ca is None or cb is None:return np.nan
    xor=np.bitwise_xor(ca,cb)
    bits=np.zeros(xor.shape,dtype=np.uint16)
    for shift in range(0,24,8):
        bits+=CENSUS_LUT[((xor>>shift)&255).astype(np.uint8)]
    return float(1.0-np.mean(bits)/24.0)

def regional_census(search_patch,template,grid=CENSUS_GRID):
    if search_patch.shape!=template.shape:return {"mean":np.nan,"min":np.nan,"quality":np.nan,"valid":False}
    h,w=template.shape
    scores=[]
    for gy in range(grid):
        y0=round(gy*h/grid);y1=round((gy+1)*h/grid)
        for gx in range(grid):
            x0=round(gx*w/grid);x1=round((gx+1)*w/grid)
            s=census_similarity(search_patch[y0:y1,x0:x1],template[y0:y1,x0:x1])
            if np.isfinite(s):scores.append(s)
    if len(scores)<5:return {"mean":np.nan,"min":np.nan,"quality":np.nan,"valid":False}
    scores=np.asarray(scores,dtype=np.float64)
    mean=float(scores.mean())
    mn=float(scores.min())
    return {"mean":mean,"min":mn,"quality":float(0.70*mean+0.30*mn),"valid":True}

def candidate_patch(search,c):
    h,w=c["template"].shape
    x0=int(round(c["x"]-w/2.0));y0=int(round(c["y"]-h/2.0))
    if x0<0 or y0<0 or x0+w>search.shape[1] or y0+h>search.shape[0]:return None
    return search[y0:y0+h,x0:x0+w]

def evaluate_census(search,c):
    patch=candidate_patch(search,c)
    if patch is None:
        c.update(census=np.nan,census_min=np.nan,census_quality=np.nan,census_valid=False)
        return c
    r=regional_census(patch,c["template"])
    c.update(census=r["mean"],census_min=r["min"],census_quality=r["quality"],census_valid=r["valid"])
    return c

def candidate_distance(a,b):
    return float(np.hypot(a["x"]-b["x"],a["y"]-b["y"]))

def deduplicate_candidates(candidates,max_count):
    candidates=sorted(candidates,key=lambda c:c["zncc"],reverse=True)
    kept=[]
    for c in candidates:
        if all(candidate_distance(c,k)>=SPATIAL_DEDUP_PX for k in kept):
            kept.append(c)
            if len(kept)>=max_count:break
    return kept

def _match_hypothesis_worker(idx,search,template):
    """Parallel NCC + peak extraction; returns (hypothesis index, peaks)."""
    response=cv2.matchTemplate(search,template,cv2.TM_CCOEFF_NORMED)
    peaks=top_peaks(response,PEAKS_PER_HYPOTHESIS)
    return idx,peaks

def generate_candidates(search,ref):
    raw=[]
    morph_cache={}
    resize_cache={}
    hypotheses=[]
    for ff in FEATURE_FACTORS:
        for scale in SCALES:
            for angle in ANGLES:
                template=transform_reference_feature(ref,ff,scale,angle,morph_cache,resize_cache)
                if template.shape[0]>search.shape[0] or template.shape[1]>search.shape[1]:continue
                # Skip zero-variance templates: OpenCV TM_CCOEFF_NORMED returns
                # an all-ones response there (templNorm<eps in templmatch.cpp).
                t=template.ravel().astype(np.float64)
                if float(((t-t.mean())**2).mean())<np.finfo(np.float64).eps:
                    continue
                hypotheses.append((ff,scale,angle,template))
    # Run NCC concurrently; restore the previous OpenCV thread count afterwards.
    results={}
    old_cv_threads=cv2.getNumThreads()
    cv2.setNumThreads(PARALLEL_OPENCV_THREADS)
    try:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures=[executor.submit(_match_hypothesis_worker,i,search,hypotheses[i][3]) for i in range(len(hypotheses))]
            for future in as_completed(futures):
                idx,peaks=future.result()
                results[idx]=peaks
    finally:
        cv2.setNumThreads(old_cv_threads)
    # Reconstruct raw candidates in production hypothesis order.
    for idx in range(len(hypotheses)):
        ff,scale,angle,template=hypotheses[idx]
        h,w=template.shape
        for score,x,y in results[idx]:
            raw.append({"x":x+w/2.0,"y":y+h/2.0,"zncc":float(score),"scale":float(scale),"angle_deg":float(angle),"feature_factor":float(ff),"template":template})
    if not raw:raise RuntimeError("No valid candidates generated.")
    return deduplicate_candidates(raw,CANDIDATE_POOL)

def select_candidate(search,candidates):
    candidates.sort(key=lambda c:c["zncc"],reverse=True)
    best=candidates[0]
    for c in candidates[:min(CENSUS_EVAL_LIMIT,len(candidates))]:evaluate_census(search,c)
    if not best.get("census_valid",False):
        for c in candidates:
            if c.get("census_valid",False):
                best=c
                break
    near=[c for c in candidates if best["zncc"]-c["zncc"]<=ZNCC_AMBIGUITY and c.get("census_valid",False)]
    if len(near)<=1:return best,"ZNCC"
    for c in near:
        c["mean_gain"]=c["census"]-best["census"]
        c["min_gain"]=c["census_min"]-best["census_min"]
    alternatives=[c for c in near if c is not best and c["mean_gain"]>=REGIONAL_MEAN_GAIN and c["min_gain"]>=REGIONAL_MIN_GAIN]
    if alternatives:return max(alternatives,key=lambda c:(c["census_quality"],c["zncc"])),"CENSUS_GATE"
    second=[c for c in candidates if best["zncc"]-c["zncc"]<=ZNCC_SECOND_CHANCE and c.get("census_valid",False)]
    for c in second:
        c["mean_gain"]=c["census"]-best["census"]
        c["min_gain"]=c["census_min"]-best["census_min"]
    second=[c for c in second if c is not best and c["mean_gain"]>=SECOND_MEAN_GAIN and c["min_gain"]>=SECOND_MIN_GAIN]
    if second:return max(second,key=lambda c:(c["census_quality"],c["zncc"])),"CENSUS_SECOND_CHANCE"
    return best,"ZNCC"

def match_pair(search,ref):
    candidates=generate_candidates(search,ref)
    selected,decision=select_candidate(search,candidates)
    # Safety gate: a near-perfect Adaptive ZNCC that beats RAW is pathological
    # (e.g. impulse noise); return the RAW baseline coordinate instead.
    if selected["zncc"]>=0.999999:
        raw=zncc_baseline(search,ref)
        if raw["score"]<selected["zncc"]:
            return {
                "x":float(raw["x"]),
                "y":float(raw["y"]),
                "zncc":float(raw["score"]),
                "census":np.nan,
                "census_min":np.nan,
                "scale":0.1,
                "rotation_deg":0.0,
                "feature_size_factor":1.0,
                "candidate_count":len(candidates),
                "decision":"ZNCC_SAFETY_GATE"
            }
    return {
        "x":float(selected["x"]),
        "y":float(selected["y"]),
        "zncc":float(selected["zncc"]),
        "census":float(selected["census"]) if np.isfinite(selected.get("census",np.nan)) else np.nan,
        "census_min":float(selected["census_min"]) if np.isfinite(selected.get("census_min",np.nan)) else np.nan,
        "scale":float(selected["scale"]),
        "rotation_deg":float(selected["angle_deg"]),
        "feature_size_factor":float(selected["feature_factor"]),
        "candidate_count":len(candidates),
        "decision":decision
    }

def dataset_path(value):
    return os.path.normpath(os.path.join(DATASET,value.replace("\\","/").strip()))

def load_manifest():
    if not os.path.exists(MANIFEST):return []
    rows=[]
    with open(MANIFEST,newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for r in reader:
            try:
                rows.append({
                    "sample":r["sample_id"].strip(),
                    "search":r["search"].strip(),
                    "reference":r["reference"].strip(),
                    "gt_x":float(r["target_center_x"]),
                    "gt_y":float(r["target_center_y"]),
                    "scale":float(r["scale"]),
                    "rotation_deg":float(r["rotation_deg"]),
                    "feature_size_factor":float(r["feature_size_factor"]),
                    "difficulty":r["difficulty"].strip()
                })
            except (KeyError,ValueError,TypeError):
                continue
    return rows

def find_gt_by_filename(search_path,ref_path):
    search_stem=os.path.splitext(os.path.basename(search_path))[0]
    ref_stem=os.path.splitext(os.path.basename(ref_path))[0]
    for suffix in ("_search","_ref","_reference"):
        if search_stem.endswith(suffix):
            search_stem=search_stem[:-len(suffix)]
        if ref_stem.endswith(suffix):
            ref_stem=ref_stem[:-len(suffix)]
    target_stems={search_stem,ref_stem}
    for row in load_manifest():
        if row["sample"] in target_stems:
            return row
    return None

def find_gt_by_pixels(search_path,ref_path):
    try:
        search_sig=image_signature(search_path)
        ref_sig=image_signature(ref_path)
    except Exception:
        return None
    for row in load_manifest():
        sp=dataset_path(row["search"])
        rp=dataset_path(row["reference"])
        if not os.path.exists(sp) or not os.path.exists(rp):continue
        try:
            if image_signature(sp)==search_sig and image_signature(rp)==ref_sig:
                return row
        except Exception:
            continue
    return None

def find_gt_by_reference_only(search_path,ref_path):
    try:
        search_img=load_raw(search_path)
        ref_img=load_raw(ref_path)
    except Exception:
        return None
    for row in load_manifest():
        sp=dataset_path(row["search"])
        rp=dataset_path(row["reference"])
        if not os.path.exists(sp) or not os.path.exists(rp):continue
        try:
            if np.array_equal(load_raw(sp),search_img) and np.array_equal(load_raw(rp),ref_img):
                return row
        except Exception:
            continue
    return None

def automatic_gt(search_path,ref_path):
    row=find_gt_by_filename(search_path,ref_path)
    if row:return row,"FILENAME"
    row=find_gt_by_pixels(search_path,ref_path)
    if row:return row,"PIXEL_HASH"
    row=find_gt_by_reference_only(search_path,ref_path)
    if row:return row,"PIXEL_EXACT"
    return None,None

def run_external(search_path,ref_path,beam_spot_nm=0.0,output_csv=None):
    gt_row,gt_method=automatic_gt(search_path,ref_path)
    if gt_row:
        print(f"Automatic GT lookup : MATCHED {gt_row['sample']} [{gt_method}]")
        gt=(gt_row["gt_x"],gt_row["gt_y"])
    else:
        print("Automatic GT lookup : NO MATCH IN LOCAL MANIFEST")
        gt=None

    search=apply_beam_spot(load_gray(search_path),beam_spot_nm)
    ref=load_gray(ref_path)

    t=time.perf_counter()
    zncc=zncc_baseline(search,ref)
    zncc_ms=(time.perf_counter()-t)*1000

    t=time.perf_counter()
    result=match_pair(search,ref)
    adaptive_ms=(time.perf_counter()-t)*1000

    print("="*64)
    print("DRIFT-SENSE | SINGLE-PAIR INFERENCE")
    print("="*64)
    print(f"Reference : {ref_path}")
    print(f"Search    : {search_path}")
    print(f"Beam spot : {beam_spot_nm:.1f} nm FWHM (evaluator-side)")
    if gt:
        print(f"Ground truth      : ({gt[0]:.2f}, {gt[1]:.2f}) px")
        print(f"GT source         : dataset manifest / {gt_row['sample']}")
    else:
        print("Ground truth      : unavailable")
    print()
    print(f"ZNCC baseline     : ({zncc['x']:.2f}, {zncc['y']:.2f}) px | score={zncc['score']:.4f} | runtime={zncc_ms:.2f} ms")
    if gt:
        print(f"ZNCC error        : {np.hypot(zncc['x']-gt[0],zncc['y']-gt[1]):.2f} px")
    cs=f"{result['census']:.4f}" if np.isfinite(result["census"]) else "N/A"
    cm=f"{result['census_min']:.4f}" if np.isfinite(result["census_min"]) else "N/A"
    print(f"Adaptive matcher  : ({result['x']:.2f}, {result['y']:.2f}) px | ZNCC={result['zncc']:.4f} | REG_CENSUS={cs} | MIN={cm}")
    print(f"Scale / rotation  : {result['scale']:.3f} / {result['rotation_deg']:+.1f}° | runtime={adaptive_ms:.2f} ms")
    print(f"Feature-size hyp. : {result['feature_size_factor']:.2f}x | decision={result['decision']}")
    if gt:
        print(f"Adaptive error    : {np.hypot(result['x']-gt[0],result['y']-gt[1]):.2f} px")
    print(f"Candidates        : {result['candidate_count']}")
    print("="*64)
    print(f"RESULT: ({result['x']:.4f}, {result['y']:.4f})")

    if output_csv:
        err=None
        if gt:
            err=float(np.hypot(result["x"]-gt[0],result["y"]-gt[1]))
        row={
            "search":search_path,
            "reference":ref_path,
            "gt_x":gt[0] if gt else "",
            "gt_y":gt[1] if gt else "",
            "pred_x":f"{result['x']:.4f}",
            "pred_y":f"{result['y']:.4f}",
            "error_px":f"{err:.4f}" if err is not None else "",
            "zncc_score":f"{result['zncc']:.4f}",
            "census_score":cs,
            "census_min":cm,
            "scale":f"{result['scale']:.3f}",
            "rotation_deg":f"{result['rotation_deg']:.1f}",
            "feature_size_factor":f"{result['feature_size_factor']:.2f}",
            "decision":result["decision"],
            "candidate_count":result["candidate_count"],
            "zncc_runtime_ms":f"{zncc_ms:.2f}",
            "adaptive_runtime_ms":f"{adaptive_ms:.2f}"
        }
        write_header=not os.path.exists(output_csv)
        with open(output_csv,"a",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"Results written to: {output_csv}")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--search",default=os.path.join(TEST_INPUT,"search.png"))
    p.add_argument("--reference",default=os.path.join(TEST_INPUT,"reference.png"))
    p.add_argument("--beam_spot_nm",type=float,default=0.0)
    p.add_argument("--output",default=None,help="Write results CSV to this path")
    a=p.parse_args()
    if not os.path.exists(a.search):raise SystemExit(f"Search image not found: {a.search}")
    if not os.path.exists(a.reference):raise SystemExit(f"Reference image not found: {a.reference}")
    run_external(a.search,a.reference,a.beam_spot_nm,a.output)
if __name__=="__main__":
    main()