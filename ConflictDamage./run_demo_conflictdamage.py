import json
import numpy as np
import pandas as pd

def generate_fake_pre_post(h=256, w=256, seed=7):
    """
    Genera dos imágenes fake (pre y post) tipo 'intensidad' en [0,1].
    - pre: textura urbana + ruido
    - post: igual, pero con 'zonas dañadas' añadidas (cambios fuertes)
    """
    rng = np.random.default_rng(seed)

    # "Textura urbana" simple: base + ruido suave + patrones
    base = rng.normal(0.5, 0.08, size=(h, w))
    stripes = (np.sin(np.linspace(0, 12*np.pi, w))[None, :] * 0.03)
    pre = np.clip(base + stripes + rng.normal(0, 0.02, size=(h, w)), 0, 1)

    post = pre.copy()

    # Añadimos 3-6 "manchas de daño" (rectángulos) con cambios de intensidad
    n_patches = rng.integers(3, 7)
    patches = []
    for _ in range(n_patches):
        ph = int(rng.integers(h//12, h//5))
        pw = int(rng.integers(w//12, w//5))
        y0 = int(rng.integers(0, h - ph))
        x0 = int(rng.integers(0, w - pw))

        delta = float(rng.uniform(0.25, 0.55))  # cambio "fuerte"
        sign = 1 if rng.random() < 0.5 else -1

        post[y0:y0+ph, x0:x0+pw] = np.clip(
            post[y0:y0+ph, x0:x0+pw] + sign * delta, 0, 1
        )
        patches.append({"x0": x0, "y0": y0, "w": pw, "h": ph, "delta": sign*delta})

    return pre, post, patches

def change_map(pre, post):
    """Mapa de cambio simple y robusto."""
    return np.abs(post - pre)

def quality_indicators(pre, post, cm):
    """
    Indicadores simples de 'calidad/fiabilidad' para el informe:
    - media de cambio, percentil 95, %pixeles con cambio alto
    """
    p95 = float(np.percentile(cm, 95))
    mean = float(cm.mean())
    high = float((cm > 0.25).mean())  # umbral simple
    return {"change_mean": mean, "change_p95": p95, "pct_high_change": high}

def tile_scores(cm, tile=32, thr=0.18):
    """
    Divide la imagen en teselas tile x tile y calcula un score por tesela:
    score = % de píxeles con cambio > thr
    """
    h, w = cm.shape
    nty = h // tile
    ntx = w // tile

    rows = []
    for ty in range(nty):
        for tx in range(ntx):
            y0 = ty * tile
            x0 = tx * tile
            block = cm[y0:y0+tile, x0:x0+tile]
            score = float((block > thr).mean())
            rows.append({
                "tile_id": f"T{ty:02d}_{tx:02d}",
                "ty": ty, "tx": tx,
                "x0": x0, "y0": y0,
                "tile_size": tile,
                "damage_score": score
            })
    df = pd.DataFrame(rows).sort_values("damage_score", ascending=False).reset_index(drop=True)
    return df

def tiles_to_geojson(df, crs_name="PIXEL_COORDS"):
    """
    GeoJSON simple en coordenadas de píxel (no georreferenciado).
    En producción, esto iría con CRS real (UTM/latlon) usando geocodificación.
    """
    features = []
    for _, r in df.iterrows():
        x0, y0, t = int(r.x0), int(r.y0), int(r.tile_size)
        # Polígono (x, y) cerrando el anillo
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
        "crs_note": crs_name,
        "features": features
    }

def main():
    pre, post, patches = generate_fake_pre_post()
    cm = change_map(pre, post)
    qi = quality_indicators(pre, post, cm)

    df = tile_scores(cm, tile=32, thr=0.18)
    top10 = df.head(10)

    geojson = tiles_to_geojson(df)

    # Outputs
    df.to_csv("outputs_conflictdamage_tiles.csv", index=False)
    with open("outputs_conflictdamage_tiles.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    with open("outputs_conflictdamage_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "quality_indicators": qi,
            "generated_patches": patches,
            "top10_tiles": top10.to_dict(orient="records")
        }, f, ensure_ascii=False, indent=2)

    print("OK ✅ Generado:")
    print("- outputs_conflictdamage_tiles.csv")
    print("- outputs_conflictdamage_tiles.geojson")
    print("- outputs_conflictdamage_report.json")
    print("\nTop 5 tiles por score:")
    print(df.head(5)[["tile_id", "damage_score"]])

if __name__ == "__main__":
    main()
