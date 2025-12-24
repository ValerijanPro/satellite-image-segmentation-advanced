
# Ovaj kod od ulaznog TIF fajla segmentacione maske (final_sam_mask_planet.tif unutar "output" foldera) pravi SHP fajl unutar "output" foldera

# takodje, odradjeni su i sledeci koraci postprocessinga:
# - filterovanje parcela manje povrsine od 3000
# - morphological opening sa kernelom 3x3
# - ponovno filterovanje parcela manje povrsine od 3000

import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from rasterio.features import rasterize

# Output paths
final_sam_mask_tif = "output/canny_roi.tif"
final_sam_shp = "output/canny.shp"

##############################################
# 9) POLYGONIZE AFTER ALL TILES
##############################################
with rasterio.open(final_sam_mask_tif) as src_sam:
    sam_arr_full = src_sam.read(1)
    sam_transform = src_sam.transform
    sam_crs = src_sam.crs

# We already thresholded each tile, so final_sam_mask is 0 or 255
# Just ensure it's strictly 0/255
sam_binary = np.where(sam_arr_full >= 128, 255, 0).astype(np.uint8)
#sam_binary = sam_arr_full
polygons = []
for geom, val in shapes(sam_binary, transform=sam_transform):
    if val == 255:
        polygons.append({"geometry": shape(geom), "value": int(val)})

gdf = gpd.GeoDataFrame(polygons, crs=sam_crs)

# --- dodaj površinu svakom poligonu --------------------------
# 1. ako CRS nije projektovani (metrički), prebaci ga – ovde pretpostavljam UTM-34N
if not gdf.crs.is_projected:
    gdf = gdf.to_crs("EPSG:32634")      # promeni EPSG po potrebi

# 2. izračunaj površinu u m² i upiši je u novo polje
gdf["area_m2"] = gdf.geometry.area      # rezultat je float (m²)
# -------------------------------------------------------------

#POSTPROCESSING

# -------------------------------------------------------------
# 10)  PRVO izbacujemo poligone < 3 000 m²,
#      PA onda čistimo uske (1-2 px) „pupčanike“
# -------------------------------------------------------------
min_area_keep = 3_000        # m²
kernel        = np.ones((3, 3), dtype="uint8")    # 3 × 3 za opening

# 10-a  Površina i prvi filter  (< 3 000 m² → OUT)
if not gdf.crs.is_projected:
    gdf = gdf.to_crs("EPSG:32634")

gdf["area_m2"] = gdf.geometry.area
gdf = gdf[gdf["area_m2"] >= min_area_keep]        # ostaju samo „krupni“

# 10-b  Rasterizuj preostale poligone (1 = poligon, 0 = pozadina)
mask_arr = rasterize(
    shapes=((geom, 1) for geom in gdf.geometry),
    out_shape=sam_binary.shape,
    transform=sam_transform,
    fill=0,
    dtype="uint8",
)

# 10-c  Morfološki opening (erode → dilate) uklanja uske trake/šiljke
from scipy.ndimage import binary_opening
mask_arr_open = binary_opening(mask_arr, structure=kernel).astype("uint8")

# 10-d  Nazad u vektor (samo pikseli koji su preživeli opening)
clean_polys = []
for geom, val in shapes(mask_arr_open, transform=sam_transform):
    if val == 1:
        clean_polys.append({"geometry": shape(geom)})

clean_gdf = gpd.GeoDataFrame(clean_polys, crs=sam_crs)

# 10-e  Još jednom izračunaj površinu i presnimi shapefile
if not clean_gdf.crs.is_projected:
    clean_gdf = clean_gdf.to_crs("EPSG:32634")

clean_gdf["area_m2"] = clean_gdf.geometry.area
clean_gdf = clean_gdf[clean_gdf["area_m2"] >= min_area_keep]        # ostaju samo „krupni“

clean_gdf["area_m2"] = clean_gdf.geometry.area
clean_gdf.to_file(final_sam_shp)
print(f"Šapfajl sa poligonima ≥ {min_area_keep} m² i bez uskih veza snimljen u → {final_sam_shp}")
