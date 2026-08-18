import os,csv,json,shutil,cv2,numpy as np,argparse

OUT_DIR="dataset"
N_SAMPLES=30
SEED=20260813
FEATURE_SIZES=(0.5,1.0,2.0)
SEARCH_SIZE=1000
REF_SIZE=1000
PARENT_SIZE=11000
ARCHITECTURE="DRAM_style_continuous_periodic"

def make_dirs():
    for d in ("search","reference","metadata"):
        os.makedirs(os.path.join(OUT_DIR,d),exist_ok=True)

def save_png(path,img):
    cv2.imwrite(path,np.clip(img,0,255).astype(np.uint8))

def correlated_field(h,w,scale,rng):
    sh=max(2,int(np.ceil(h/scale)))
    sw=max(2,int(np.ceil(w/scale)))
    field=rng.normal(0,1,(sh,sw)).astype(np.float32)
    field=cv2.resize(field,(w,h),interpolation=cv2.INTER_CUBIC)
    field/=max(float(field.std()),1e-6)
    return field

def generate_parent(seed,fs):
    r=np.random.default_rng(seed)
    H=W=PARENT_SIZE
    img=np.full((H,W),32,np.float32)

    pitch=400.0
    wl=max(2.0,48.0*fs)
    bl=max(2.0,42.0*fs)

    nx=W//int(pitch)+1
    ny=H//int(pitch)+1
    xj=r.normal(0,10,nx)
    yj=r.normal(0,10,ny)
    xs=np.arange(nx)*pitch+xj
    ys=np.arange(ny)*pitch+yj

    for y in ys:
        width=max(2.0,wl+r.normal(0,4*fs))
        val=205+r.normal(0,4)
        y1=max(0,int(round(y-width/2)))
        y2=min(H,int(round(y+width/2)))
        if y2>y1:
            img[y1:y2,:]=val

    for x in xs:
        width=max(2.0,bl+r.normal(0,4*fs))
        val=185+r.normal(0,4)
        x1=max(0,int(round(x-width/2)))
        x2=min(W,int(round(x+width/2)))
        if x2>x1:
            img[:,x1:x2]=val

    img+=correlated_field(H,W,300,r)*5.0

    yy,xx=np.meshgrid(np.arange(len(ys)),np.arange(len(xs)),indexing="ij")
    local_phase=np.sin(xx*0.73+yy*0.41)+np.cos(xx*0.31-yy*0.67)

    for j,y in enumerate(ys):
        for i,x in enumerate(xs):
            cr=np.random.default_rng(seed+i*1009+j*9176)
            dx=cr.normal(0,8)
            dy=cr.normal(0,8)

            size=max(
                4.0,
                (62+cr.normal(0,5))*fs+local_phase[j,i]*3*fs
            )

            intensity=225+cr.normal(0,7)+local_phase[j,i]*4

            if cr.random()<0.02:
                size*=0.7
                intensity*=0.45

            cx=int(round(x+dx))
            cy=int(round(y+dy))
            half=max(2,int(round(size/2)))

            x1=max(0,cx-half)
            x2=min(W,cx+half)
            y1=max(0,cy-half)
            y2=min(H,cy+half)

            if x2>x1 and y2>y1:
                img[y1:y2,x1:x2]=intensity

    return img

def edge_brighten(img,amount=0.6,sigma=1.2):
    """Unsharp-mask edge brightening, mimicking SEM secondary-electron edge
    contrast: brighter signal along feature edges than in flat regions."""
    blur=cv2.GaussianBlur(img,(0,0),sigmaX=sigma,sigmaY=sigma)
    return np.clip(img+amount*(img-blur),0,255)

def degrade(img,r,reference=False):
    out=img.copy()
    sigma=r.uniform(0.45,0.9) if reference else r.uniform(0.7,1.2)
    out=cv2.GaussianBlur(out,(0,0),sigma)
    out*=r.uniform(0.92,1.08)
    out+=r.uniform(-6,6)
    # Search images carry MORE detector noise than references (test data is
    # more noisy than training data). Independent noise per image.
    out+=r.normal(0,r.uniform(2,4) if reference else r.uniform(3,6),out.shape).astype(np.float32)
    gamma=r.uniform(0.94,1.06)
    out=np.clip(out/255.0,0,1)
    out=(out**gamma)*255.0
    return edge_brighten(np.clip(out,0,255))

def rotate(img,angle):
    if abs(angle)<1e-6:
        return img
    h,w=img.shape
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(
        img,M,(w,h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

def crop(img,cx,cy,size):
    half=size//2
    x1=int(round(cx-half))
    y1=int(round(cy-half))
    x1=max(0,min(x1,img.shape[1]-size))
    y1=max(0,min(y1,img.shape[0]-size))
    x2=x1+size
    y2=y1+size
    return img[y1:y2,x1:x2],x1,y1

def difficulty(i):
    if i<6:return "easy"
    if i<15:return "moderate"
    if i<23:return "ambiguous"
    return "highly_repetitive"

def make_schedule():
    schedule=[]
    base=N_SAMPLES//len(FEATURE_SIZES)
    remainder=N_SAMPLES%len(FEATURE_SIZES)

    for i,fs in enumerate(FEATURE_SIZES):
        count=base+(1 if i<remainder else 0)
        schedule.extend([fs]*count)

    np.random.default_rng(SEED+424242).shuffle(schedule)
    return schedule

def make_sample(i,parent,fs):
    seed=SEED+i*7919
    r=np.random.default_rng(seed)
    diff=difficulty(i)

    scale=float(r.choice([9.0,10.0,11.0]))
    angle=float(r.choice([
        0.0,0.0,0.0,
        1.0,-1.0,1.5,-1.5,2.0,-2.0
    ]))

    extent=int(round(SEARCH_SIZE*scale))

    if extent>PARENT_SIZE:
        raise RuntimeError(
            f"Search extent {extent} exceeds parent {PARENT_SIZE}."
        )

    max_sx=PARENT_SIZE-extent
    max_sy=PARENT_SIZE-extent
    sx=int(r.integers(0,max_sx+1)) if max_sx>0 else 0
    sy=int(r.integers(0,max_sy+1)) if max_sy>0 else 0

    target_margin=1000

    if extent<=2*target_margin:
        raise RuntimeError(
            f"Search extent {extent} too small for target margin."
        )

    target_x=int(r.integers(
        sx+target_margin,
        sx+extent-target_margin+1
    ))
    target_y=int(r.integers(
        sy+target_margin,
        sy+extent-target_margin+1
    ))

    search_parent=parent[sy:sy+extent,sx:sx+extent]
    search=cv2.resize(
        search_parent,
        (SEARCH_SIZE,SEARCH_SIZE),
        interpolation=cv2.INTER_AREA
    )

    gt_x=(target_x-sx)/scale
    gt_y=(target_y-sy)/scale

    reference,rx,ry=crop(parent,target_x,target_y,REF_SIZE)

    search=degrade(search,r,False)
    reference=degrade(reference,r,True)
    reference=rotate(reference,angle)

    name=f"sample_{i:03d}"

    search_path=os.path.join(
        OUT_DIR,"search",f"{name}_search.png"
    )
    ref_path=os.path.join(
        OUT_DIR,"reference",f"{name}_ref.png"
    )
    meta_path=os.path.join(
        OUT_DIR,"metadata",f"{name}_metadata.json"
    )

    save_png(search_path,search)
    save_png(ref_path,reference)

    meta={
        "sample_id":name,
        "seed":seed,
        "architecture":ARCHITECTURE,
        "difficulty":diff,
        "scale":scale,
        "rotation_deg":angle,
        "feature_size_factor":float(fs),
        "feature_size_definition":
            "Critical line/contact dimensions vary while lattice pitch remains fixed.",
        "nominal_pitch_parent_px":400.0,
        "word_line_width_parent_px":48.0*fs,
        "bit_line_width_parent_px":42.0*fs,
        "contact_nominal_size_parent_px":62.0*fs,
        "pitch_preserved":True,
        "search_width":SEARCH_SIZE,
        "search_height":SEARCH_SIZE,
        "reference_width":REF_SIZE,
        "reference_height":REF_SIZE,
        "parent_width":PARENT_SIZE,
        "parent_height":PARENT_SIZE,
        "target_center_x":float(gt_x),
        "target_center_y":float(gt_y),
        "target_parent_x":target_x,
        "target_parent_y":target_y,
        "search_parent_x":sx,
        "search_parent_y":sy,
        "search_parent_size":extent,
        "reference_parent_x":rx,
        "reference_parent_y":ry,
        "line_width_variation":True,
        "contact_variation":True,
        "local_jitter":True,
        "weak_missing_features":True,
        "independent_search_degradation":True,
        "independent_reference_degradation":True,
        "ground_truth_source":"parent_geometry"
    }

    with open(meta_path,"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)

    return {
        "sample_id":name,
        "search":f"search/{name}_search.png",
        "reference":f"reference/{name}_ref.png",
        "metadata":f"metadata/{name}_metadata.json",
        "seed":seed,
        "architecture":ARCHITECTURE,
        "difficulty":diff,
        "scale":scale,
        "rotation_deg":angle,
        "feature_size_factor":float(fs),
        "target_center_x":gt_x,
        "target_center_y":gt_y
    }

def write_manifests(rows):
    rows=sorted(rows,key=lambda x:x["sample_id"])

    csv_path=os.path.join(OUT_DIR,"manifest.csv")
    json_path=os.path.join(OUT_DIR,"manifest.json")

    with open(csv_path,"w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    counts={}
    for row in rows:
        key=f"{row['feature_size_factor']:.1f}"
        counts[key]=counts.get(key,0)+1

    manifest={
        "dataset":"DRIFT-SENSE V2.1",
        "version":"2.1",
        "seed":SEED,
        "samples":len(rows),
        "search_size":SEARCH_SIZE,
        "reference_size":REF_SIZE,
        "parent_size":PARENT_SIZE,
        "scales":[9.0,10.0,11.0],
        "rotation_range_deg":[-2.0,2.0],
        "feature_size_factors":[0.5,1.0,2.0],
        "feature_size_counts":counts,
        "feature_size_definition":
            "Critical line/contact dimensions vary while lattice pitch remains fixed.",
        "architecture":ARCHITECTURE,
        "ground_truth_source":"parent_geometry",
        "samples_data":rows
    }

    with open(json_path,"w",encoding="utf-8") as f:
        json.dump(manifest,f,indent=2)

def main():
    global N_SAMPLES, OUT_DIR

    parser=argparse.ArgumentParser(
        description="DRIFT-SENSE V2.1 DRAM-style dataset generator"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=N_SAMPLES,
        help="number of reference+search pairs to generate"
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=None,
        help="alias for --samples (competition guideline wording)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="output directory (default: ./dataset)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow overwriting a non-empty output directory"
    )
    args=parser.parse_args()

    count=args.pairs if args.pairs is not None else args.samples
    if count<=0:
        raise SystemExit("--samples/--pairs must be > 0")

    N_SAMPLES=count
    if args.output:
        OUT_DIR=args.output

    if os.path.exists(OUT_DIR) and os.listdir(OUT_DIR) and not args.force:
        raise SystemExit(
            f"Output directory '{OUT_DIR}' exists and is not empty. "
            f"Pass --force to overwrite it (this DELETES its contents)."
        )
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    make_dirs()
    schedule=make_schedule()

    print("DRIFT-SENSE | Dataset Generator V2.1")
    print(f"Architecture: {ARCHITECTURE}")
    print(f"Output dir  : {OUT_DIR}")
    print(f"Samples     : {N_SAMPLES}")

    for fs in FEATURE_SIZES:
        print(
            f"Feature {fs:.1f}x: "
            f"{schedule.count(fs)} samples"
        )

    rows=[]

    for fs in FEATURE_SIZES:
        indices=[
            i for i,factor in enumerate(schedule)
            if factor==fs
        ]

        if not indices:
            continue

        print(f"\nGenerating {fs:.1f}x parent...")
        parent=generate_parent(SEED,fs)

        for i in indices:
            row=make_sample(i,parent,fs)
            rows.append(row)

            print(
                f"{row['sample_id']} | "
                f"{row['difficulty']} | "
                f"s={row['scale']:.1f} | "
                f"r={row['rotation_deg']:+.1f}° | "
                f"feature={fs:.1f}x | "
                f"GT=({row['target_center_x']:.1f},"
                f"{row['target_center_y']:.1f})"
            )

        del parent

    write_manifests(rows)

    print("\nDataset complete.")
    print(f"Search:    {OUT_DIR}\\search")
    print(f"Reference: {OUT_DIR}\\reference")
    print(f"Metadata:  {OUT_DIR}\\metadata")
    print(f"Manifest:  {OUT_DIR}\\manifest.csv")
    print(f"Manifest:  {OUT_DIR}\\manifest.json")

if __name__=="__main__":
    main()