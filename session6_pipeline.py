#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CDIO1 – Session 6: Detect waterbodies and estimate coastline
(Standalone script)
"""
import argparse, glob, json, os, re
from typing import List, Optional, Sequence, Tuple
import numpy as np, rasterio
from rasterio.mask import mask as rio_mask
from scipy.ndimage import convolve
import matplotlib

def parse_date_from_filename(path: str) -> Optional[str]:
    name = os.path.basename(path)
    for pat in (r"(20\d{2}[-_]\d{2}[-_]\d{2})", r"(20\d{2}\d{2}\d{2})"):
        m = re.search(pat, name)
        if m:
            raw = m.group(1).replace("_","-")
            if len(raw)==8: return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"
            return raw
    return None

def detect_waterbody(ndwi: np.ndarray, threshold: float=0.0)->np.ndarray:
    ndwi = np.array(ndwi, copy=False, dtype=np.float32)
    ndwi = np.nan_to_num(ndwi, nan=-1e9, posinf=-1e9, neginf=-1e9)
    return (ndwi>threshold).astype(np.uint8)

def test_detect_waterbody():
    a = np.array([[0.3,-0.2,0.0],[np.nan,0.01,-1.5],[np.inf,-np.inf,0.5]],dtype=np.float32)
    wb = detect_waterbody(a)
    exp= np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.uint8)
    assert np.array_equal(wb, exp)

def coastline_mask(waterbody: np.ndarray, connectivity:int=8)->np.ndarray:
    if connectivity==8:
        kernel=np.array([[1,1,1],[1,0,1],[1,1,1]],dtype=np.uint8); maxn=8
    elif connectivity==4:
        kernel=np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.uint8); maxn=4
    else:
        raise ValueError("connectivity debe ser 4 u 8")
    neigh = convolve(waterbody.astype(np.uint8), kernel, mode="constant", cval=0)
    return np.logical_and(waterbody==1, neigh<maxn).astype(np.uint8)

def coastline_points(mask_arr: np.ndarray, transform, date_str: Optional[str]):
    ys, xs = np.where(mask_arr==1)
    out=[]; xy = rasterio.transform.xy
    for r,c in zip(ys,xs):
        x,y = xy(transform, r, c, offset="center")
        out.append((float(x),float(y),date_str))
    return out

def export_points_geojson(points, out_path:str, crs):
    feats=[{"type":"Feature","properties":({"date":d} if d else {}),
            "geometry":{"type":"Point","coordinates":[x,y]}} for x,y,d in points]
    gj={"type":"FeatureCollection","features":feats}
    if crs: gj["crs"]={"type":"name","properties":{"name":str(crs)}}
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(gj,f,ensure_ascii=False,indent=2)

def export_points_csv(points, out_path:str):
    import csv
    with open(out_path,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["X","Y","Date"])
        for x,y,d in points: w.writerow([x,y,d if d else ""])

def process_single_ndwi(ndwi_path:str,outdir:str,threshold:float=0.0,connectivity_list:Sequence[int]=(8,),aoi_path:Optional[str]=None,prefix:Optional[str]=None)->List[str]:
    os.makedirs(outdir,exist_ok=True)
    date_str=parse_date_from_filename(ndwi_path)
    base=os.path.splitext(os.path.basename(ndwi_path))[0]
    tag=f"{prefix}_" if prefix else ""
    outs=[]
    with rasterio.open(ndwi_path) as src:
        if aoi_path:
            with open(aoi_path,"r",encoding="utf-8") as f: gj=json.load(f)
            feats = gj["features"] if gj.get("type")=="FeatureCollection" else [gj]
            ndwi_arr, out_transform = rio_mask(src, [ft["geometry"] for ft in feats], crop=True, filled=True, nodata=0)
            ndwi = ndwi_arr[0].astype(np.float32)
            profile=src.profile.copy()
            profile.update(height=ndwi.shape[0], width=ndwi.shape[1], transform=out_transform)
        else:
            ndwi = src.read(1).astype(np.float32)
            profile=src.profile.copy()
        crs=src.crs
    water=detect_waterbody(ndwi,threshold=threshold)
    wb_tif=os.path.join(outdir,f"{tag}{base}_waterbody.tif")
    prof=profile.copy(); prof.update(count=1,dtype=rasterio.uint8,compress="DEFLATE",nodata=0)
    with rasterio.open(wb_tif,"w",**prof) as dst: dst.write(water,1)
    outs.append(wb_tif)
    for conn in connectivity_list:
        coast=coastline_mask(water,connectivity=conn)
        cl_tif=os.path.join(outdir,f"{tag}{base}_coastline_conn{conn}.tif")
        with rasterio.open(cl_tif,"w",**prof) as dst: dst.write(coast,1)
        outs.append(cl_tif)
        pts=coastline_points(coast, prof["transform"], date_str)
        cl_geo=os.path.join(outdir,f"{tag}{base}_coastline_points_conn{conn}.geojson")
        cl_csv=os.path.join(outdir,f"{tag}{base}_coastline_points_conn{conn}.csv")
        export_points_geojson(pts, cl_geo, crs); export_points_csv(pts, cl_csv)
        outs += [cl_geo, cl_csv]
    return outs

def process_folder(ndwi_folder:str,outdir:str,pattern:str="*_ndwi.tif",threshold:float=0.0,connectivity_list:Sequence[int]=(8,),aoi_path:Optional[str]=None,prefix:Optional[str]=None):
    paths=sorted(glob.glob(os.path.join(ndwi_folder,pattern)))
    if not paths: raise FileNotFoundError(f"No NDWI en {ndwi_folder!r} con patrón {pattern!r}")
    all_outs=[]
    for p in paths:
        print("Procesando:",p)
        all_outs.append(process_single_ndwi(p,outdir,threshold,connectivity_list,aoi_path,prefix))
    return all_outs

def build_argparser():
    ap=argparse.ArgumentParser(description="CDIO1 S6: NDWI -> Waterbody -> Coastline -> GeoJSON/CSV")
    g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ndwi", help="Ruta a un GeoTIFF de NDWI")
    g.add_argument("--ndwi-folder", help="Carpeta con GeoTIFFs NDWI")
    ap.add_argument("--pattern", default="*_ndwi.tif")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--connectivity", type=int, nargs="+", default=[8], choices=[4,8])
    ap.add_argument("--aoi", help="AOI GeoJSON (opcional)")
    ap.add_argument("--prefix")
    ap.add_argument("--run-tests", action="store_true")
    return ap

def main():
    args=build_argparser().parse_args()
    if args.run_tests:
        test_detect_waterbody(); print("✅ Tests OK"); return
    if args.ndwi:
        outs=process_single_ndwi(args.ndwi,args.outdir,args.threshold,args.connectivity,args.aoi,args.prefix)
        print("✅ Generado:"); [print(" ",p) for p in outs]
    else:
        all_outs=process_folder(args.ndwi_folder,args.outdir,args.pattern,args.threshold,args.connectivity,args.aoi,args.prefix)
        print("✅ Lote terminado."); [print(" ",p) for outs in all_outs for p in outs]

if __name__=="__main__":
    main()
