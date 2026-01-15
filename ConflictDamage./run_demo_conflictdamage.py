import os
import json
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_gray_png(arr01: np.ndarray, path: str):
    arr255 = (np.clip(arr01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(arr255, mode="L").save(path)

def load_image_rgb(path: str):
    return Image.open(path).convert("RGB")

def rgb_to_gray01(img_rgb: Image.Image):
    arr = np.asarray(img_rgb).astype(np.float32) / 255.0
    return arr.mean(axis=2)

def change_map(pre, post):
    return np.abs(post - pre)

def quality_indicators(cm):
    return {
        "change_mean": float(cm.mean()),
        "change_p95": float(np.percentile(cm, 95)),
        "pct_high_change_thr025": float((cm > 0.25).mean())
    }

def tile_scores(cm, tile=64, thr=0.22):
    h, w = cm.shape
    nty = h // tile
    ntx = w // tile

    rows = []
    for ty in range(nty):
        for tx in range(ntx):
            y0 = ty * tile
            x0 = tx * tile
            block = cm[y0:y0+tile, x0:x0+tile]

            # Score: % de píxeles con cambio mayor que umbral
            score = float((block > thr).mean())

            rows.append({
                "tile_id": f"T{ty:02d}_{tx:02d}",
                "ty": ty, "tx": tx,
                "x0": int(x0), "y0": int(y0),
                "tile_size": int(tile),
                "damage_score": score
            })

    df = pd.DataFrame(rows).sort_values("damage_score", ascending=False).reset_index(drop=True)
    return df

def tiles_to_geojson(df, crs_note="PIXEL_COORDS"):
    features = []
    for _, r in df.iterrows():
        x0, y0, t = int(r.x0), int(r.y0), int(r.tile_size)
        poly = [[
            [x0, y0],
            [x0 + t, y0],
            [x0 + t, y0 + t],
            [x0, y0 + t],
            [x0, y0],
        ]]
        features.append({
            "type": "Feature",
            "properties": {
                "tile_id": r.tile_id,
                "damage_score": float(r.damage_score)
            },
            "geometry": {"type": "Polygon", "coordinates": poly}
        })

    return {
        "type": "FeatureCollection",
        "name": "conflictdamage_tiles",
        "crs_note": crs_note,
        "features": features
    }

def draw_grid_on_image(img_rgb: Image.Image, tile: int, out_path: str):
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Líneas verticales
    for x in range(0, w, tile):
        draw.line([(x, 0), (x, h)], width=1)
    # Líneas horizontales
    for y in range(0, h, tile):
        draw.line([(0, y), (w, y)], width=1)

    img.save(out_path)

def draw_top_tiles_overlay(img_rgb: Image.Image, df: pd.DataFrame, k: int, out_path: str):
    """
    Dibuja rectángulos en las K teselas con más score.
    (Para que el profe vea qué zonas detecta como más dañadas.)
    """
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)

    top = df.head(k)
    for i, r in enumerate(top.itertuples(index=False), start=1):
        x0, y0, t = int(r.x0), int(r.y0), int(r.tile_size)
        draw.rectangle([x0, y0, x0+t, y0+t], width=3)
        # etiqueta simple
        draw.text((x0+3, y0+3), f"{i}:{r.damage_score:.2f}")

    img.save(out_path)

def tile_score_map_image(df: pd.DataFrame, width: int, height: int, tile: int):
    """
    Crea una imagen gris donde cada tesela tiene intensidad proporcional al score.
    (0=negro, 1=blanco)
    """
    img = np.zeros((height, width), dtype=np.float32)

    for r in df.itertuples(index=False):
        x0, y0, t = int(r.x0), int(r.y0), int(r.tile_size)
        val = float(r.damage_score)
        img[y0:y0+t, x0:x0+t] = val

    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre", type=str, required=True, help="Imagen PRE (jpg/png)")
    parser.add_argument("--post", type=str, required=True, help="Imagen POST (jpg/png)")
    parser.add_argument("--tile", type=int, default=64, help="Tamaño tesela (pixeles)")
    parser.add_argument("--thr", type=float, default=0.22, help="Umbral cambio para score")
    parser.add_argument("--topk", type=int, default=10, help="Teselas top para marcar")
    parser.add_argument("--outdir", type=str, default="outputs", help="Carpeta salida")
    args = parser.parse_args()

    ensure_dir(args.outdir)

    # Cargar imágenes RGB reales
    pre_rgb = load_image_rgb(args.pre)
    post_rgb = load_image_rgb(args.post)

    # Asegurar mismo tamaño: reescalamos POST al tamaño de PRE
    W, H = pre_rgb.size
    post_rgb = post_rgb.resize((W, H), Image.BILINEAR)

    pre = rgb_to_gray01(pre_rgb)
    post = rgb_to_gray01(post_rgb)

    cm = change_map(pre, post)
    qi = quality_indicators(cm)

    df = tile_scores(cm, tile=args.tile, thr=args.thr)
    top10 = df.head(10)

    # Outputs “docs”
    df.to_csv(os.path.join(args.outdir, "tiles.csv"), index=False)
    with open(os.path.join(args.outdir, "tiles.geojson"), "w", encoding="utf-8") as f:
        json.dump(tiles_to_geojson(df), f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "mode": "real_images",
            "pre_image": args.pre,
            "post_image": args.post,
            "image_size": {"width": int(W), "height": int(H)},
            "params": {"tile": args.tile, "thr": args.thr, "topk": args.topk},
            "quality_indicators": qi,
            "top10_tiles": top10.to_dict(orient="records")
        }, f, ensure_ascii=False, indent=2)

    # Imágenes que ayudan al profe a entender el cálculo
    save_gray_png(pre, os.path.join(args.outdir, "pre_gray.png"))
    save_gray_png(post, os.path.join(args.outdir, "post_gray.png"))

    cm_norm = cm / (cm.max() + 1e-9)
    save_gray_png(cm_norm, os.path.join(args.outdir, "change_map.png"))
    mask = (cm > args.thr).astype(np.float32)
    save_gray_png(mask, os.path.join(args.outdir, "change_mask.png"))

    # Cuadrículas
    draw_grid_on_image(pre_rgb, args.tile, os.path.join(args.outdir, "pre_grid.png"))
    draw_grid_on_image(post_rgb, args.tile, os.path.join(args.outdir, "post_grid.png"))

    # Marcar top K teselas sobre el POST (lo más convincente)
    draw_top_tiles_overlay(post_rgb, df, args.topk, os.path.join(args.outdir, "post_top_tiles.png"))

    # Imagen de “score por teselas”
    score_img = tile_score_map_image(df, W, H, args.tile)
    save_gray_png(score_img, os.path.join(args.outdir, "tile_score_map.png"))

    print("OK ✅ Generado en:", args.outdir)
    print("- tiles.csv / tiles.geojson / report.json")
    print("- pre_grid.png / post_grid.png")
    print("- post_top_tiles.png (top teselas marcadas)")
    print("- tile_score_map.png (score por teselas)")
    print("\nTop 5 tiles por score:")
    print(df.head(5)[["tile_id", "damage_score"]])

if __name__ == "__main__":
    main()
