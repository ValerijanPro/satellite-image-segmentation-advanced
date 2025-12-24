# tif_edges_to_regions.py
import os
import sys
import json
import numpy as np
import rasterio
from rasterio.features import shapes as rs_shapes
from scipy.ndimage import binary_dilation, binary_closing, binary_fill_holes
import fiona
from fiona.crs import from_string

IN_TIF      = "output/canny_roi3.tif"       # 0/255 raster ivica (Canny posle maske)
OUT_SHP     = "output/canny_regions3.shp"   # poligoni "između" ivica
OUT_GEOJSON = "output/canny_regions3.geojson"

# Parametri za zatvaranje rupa u ivicama:
DILATE_ITERS = 1     # povećaj ako su ivice “prekinute” (npr. 2-3)
CLOSE_ITERS  = 0     # opcionalno, dodatno zatvaranje (0=isključen)

def main():
    if not os.path.exists(IN_TIF):
        print(f"[ERROR] Ne postoji ulazni raster: {IN_TIF}")
        sys.exit(1)

    with rasterio.open(IN_TIF) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    # 1) Binarno: 1 = ivica, 0 = sve ostalo
    edges = (arr >= 128).astype(np.uint8)
    nz = int(edges.sum())
    print(f"[INFO] Piksel-ivica: {nz}")
    if nz == 0:
        print("[WARN] Nema ivica u rasteru.")
        sys.exit(0)

    # 2) Opcionalno zadebljaj ivice da zatvoriš sitne prekide
    if DILATE_ITERS > 0:
        edges = binary_dilation(edges, iterations=DILATE_ITERS).astype(np.uint8)

    # 2b) Opcionalno closing (dilation -> erosion) za krpljenje praznina
    if CLOSE_ITERS > 0:
        edges = binary_closing(edges, iterations=CLOSE_ITERS).astype(np.uint8)

    # 3) Popuni rupe tretirajući ivice kao "zidove"
    #    fill_holes radi nad "True" kao zidovima, pa mu damo edges==1
    filled = binary_fill_holes(edges.astype(bool)).astype(np.uint8)

    # 4) Unutrašnjost = popunjeno − ivice
    regions = np.clip(filled - edges, 0, 1).astype(np.uint8)
    nreg = int(regions.sum())
    print(f"[INFO] Piksel-unutrašnjosti: {nreg}")
    if nreg == 0:
        print("[WARN] Nema zatvorenih regiona (verovatno su ivice previše “šuplje”). "
              "Povećaj DILATE_ITERS i/ili CLOSE_ITERS.")
        sys.exit(0)

    # 5) Poligonizacija samo gde je 1
    def polygonize(binary):
        for geom, val in rs_shapes(binary, transform=transform, connectivity=8):
            if val != 1:
                continue
            gtype = geom.get("type", "")
            if gtype in ("Polygon", "MultiPolygon"):
                yield geom

    geoms = list(polygonize(regions))
    print(f"[INFO] Broj poligona za upis: {len(geoms)}")
    if not geoms:
        print("[WARN] Nema poligona za upis.")
        sys.exit(0)

    # 6) Upis (Shapefile + GeoJSON)
    os.makedirs(os.path.dirname(OUT_SHP), exist_ok=True)
    schema = {"geometry": "Polygon", "properties": {"value": "int"}}
    crs_fiona = from_string(crs.to_string()) if crs else None

    # Shapefile
    count = 0
    with fiona.open(OUT_SHP, "w", driver="ESRI Shapefile", schema=schema, crs=crs_fiona) as sink:
        for geom in geoms:
            try:
                sink.write({"geometry": geom, "properties": {"value": 1}})
                count += 1
            except Exception:
                # preskoči degenerisane geometr.
                continue
    print(f"[OK] Shapefile: {OUT_SHP} (upisano poligona: {count})")

    # GeoJSON (korisno za brzu proveru)
    feats = [{"type": "Feature", "geometry": g, "properties": {"value": 1}} for g in geoms]
    with open(OUT_GEOJSON, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    print(f"[OK] GeoJSON:   {OUT_GEOJSON}")

if __name__ == "__main__":
    main()
