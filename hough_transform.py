import rasterio
import numpy as np
from skimage.feature import canny
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line
import geopandas as gpd
from shapely.geometry import LineString, shape
from shapely.ops import unary_union, linemerge
from scipy.ndimage import binary_closing, binary_fill_holes, binary_dilation
from skimage.measure import label
from rasterio.features import rasterize, shapes
import matplotlib.pyplot as plt
import os

# parametri
in_mask_tif     = "output/final_sam_mask_planet.tif"
out_shp         = "output/final_sam_parcels1.shp"
sigma           = 1.0

# podešavanje za Hough:
hough_threshold = 10   # granica glasova unutar akumulatora, da nesto bude smatrano linijom (vise = stroze)
line_length     = 10  # minimalna duzina segmenta u pikselima
line_gap        = 5    # maksimalna rupa između segmenata koju ce popuniti hough

output_png = os.path.join("output", "skeleton_for_hough.png")
os.makedirs(os.path.dirname(output_png), exist_ok=True)

# 1) Učitaj masku (0/255)
with rasterio.open(in_mask_tif) as src:
    mask      = src.read(1)
    transform = src.transform
    crs       = src.crs
    H, W      = mask.shape

# 2) Canny ivice
img_float = (mask.astype(np.float32) / 255.0)
edges     = canny(img_float, sigma=sigma)

# 3) Skeletonizacija
skel = skeletonize(edges)

# 4) Kratko closing na skeletonu (da uklonimo vrlo tanke artefakte)
closing_struct_skel = np.ones((3,3), dtype=np.uint8)
skel_closed = binary_closing(skel, structure=closing_struct_skel)

# 5) (Opcionalno) Ukloni male fragmente <20 piksela
labels = label(skel_closed)
skel_filtered = np.zeros_like(skel_closed, dtype=bool)
for lbl in np.unique(labels):
    if lbl == 0:
        continue
    size = np.sum(labels == lbl)
    if size > 20:
        skel_filtered[labels == lbl] = True

# 6) Snimi taj „očišćeni” skeleton, da proveriš
plt.imsave(output_png, skel_filtered, cmap='gray')
print(f"Skeleton za Hough je sačuvan kao PNG u:\n  → {output_png}")

# 7) Probabilistički Hough na skel_filtered
segments = probabilistic_hough_line(
    skel_filtered,
    threshold=hough_threshold,
    line_length=line_length,
    line_gap=line_gap
)

# 8) Konvertuj svaki segment u LineString (u prave koordinate)
lines = []
for (p0, p1) in segments:
    x0, y0 = rasterio.transform.xy(transform, p0[0], p0[1])
    x1, y1 = rasterio.transform.xy(transform, p1[0], p1[1])
    lines.append(LineString([(x0, y0), (x1, y1)]))

# 9) Spoji kolinearne segmente
merged = linemerge(unary_union(lines))
if merged.geom_type == 'LineString':
    merged_lines = [merged]
else:
    merged_lines = list(merged.geoms)

# ──────────────────────────────────────────────────────────────────────────
# 10) Rasterizuj spojene linije nazad na grid, da bi ih moglo se dilatirati
# ──────────────────────────────────────────────────────────────────────────
shapes_for_raster = [(geom, 1) for geom in merged_lines]
lines_raster = rasterize(
    shapes_for_raster,
    out_shape=(H, W),
    transform=transform,
    fill=0,
    dtype="uint8"
)

# 11) Debljanje linija (dilatacija) da se spoje, npr. 5×5 strukt element
dilate_selem = np.ones((5,5), dtype=np.uint8)
lines_dilated = binary_dilation(lines_raster == 1, structure=dilate_selem)

# 12) Zatvaranje pukotina na dilatiranim linijama (da budu potpune ivice)
closing_struct = np.ones((5,5), dtype=np.uint8)
lines_closed = binary_closing(lines_dilated, structure=closing_struct)

# 13) Inverzija: True = unutrašnji delovi između ivica (parcele)
parcels_mask = np.logical_not(lines_closed)

# 14) Flood-fill rupa unutar parcela
parcels_filled = binary_fill_holes(parcels_mask)

# 15) Još jedno closing da uklonimo tanke mostiće
parcels_clean = binary_closing(parcels_filled, structure=closing_struct)

# 16) Polygonize: izraštaj poligone iz tih belih (1) regiona
polygons = []
for geom_dict, val in shapes(parcels_clean.astype(np.uint8), transform=transform):
    if val == 1:
        polygons.append({"geometry": shape(geom_dict)})

gdf_parcels = gpd.GeoDataFrame(polygons, crs=crs)

# 17) Reprojekcija + filtriranje malih (<3000 m²)
if not gdf_parcels.crs.is_projected:
    gdf_parcels = gdf_parcels.to_crs("EPSG:32634")
gdf_parcels["area_m2"] = gdf_parcels.geometry.area
gdf_parcels = gdf_parcels[gdf_parcels["area_m2"] >= 3000]

# 18) Čuvanje shapefile-a
os.makedirs(os.path.dirname(out_shp), exist_ok=True)
gdf_parcels.to_file(out_shp)

print("Gotovi poligoni (parcele) između Hough linija pohranjeni u:")
print("  →", out_shp)
