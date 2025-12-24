import os
import random
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject, Resampling, transform_bounds
from skimage.feature import canny
from skimage.io import imsave

# ---- Ulazi ----
image_input_path = "composite.tif"
mask_input_path  = "../ParcelaMaska.tif"

# Veličina ROI (pikseli). Lokacija će se birati automatski tako da bude unutar maske.
ROI_WIDTH   = 1024
ROI_HEIGHT  = 1024

# Canny parametri
CANNY_SIGMA = 4
CANNY_LOW   = None
CANNY_HIGH  = None

# Pragovi za biranje “dobrog” ROI-a
MIN_MASK_COVERAGE = 0.10   # tražimo bar 10% pokrivenosti maskom u ROI-u
MIN_VALID_PIXELS  = 0.05   # tražimo bar 5% validnih piksela u slici (NoData ignorisan)
MAX_TRIES = 300            # max broj pokušaja nasumičnih prozora

# Izlazi
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)
output_tif = os.path.join(out_dir, "canny_roi3.tif")
output_png = os.path.join(out_dir, "canny_roi3.png")

def robust_gray_from_bands(arr_masked):
    """ MaskedArray (bands,H,W) -> grayscale [0,1], robustno (2–98 pct), ignoriše NoData i nule. """
    bands = min(arr_masked.shape[0], 3)
    stretched = []
    for b in range(bands):
        band = arr_masked[b]
        data = band.compressed()
        data = data[data != 0]
        if data.size < 10:
            stretched.append(np.zeros(band.shape, dtype=np.float32))
            continue
        p2, p98 = np.percentile(data, [2, 98])
        if p98 > p2:
            bf = band.astype(np.float32).filled(0.0)
            bf = (bf - p2) / (p98 - p2)
            bf = np.clip(bf, 0, 1)
        else:
            bf = np.zeros(band.shape, dtype=np.float32)
        stretched.append(bf)
    return np.mean(np.stack(stretched, axis=0), axis=0).astype(np.float32)

def read_window_masked(src, col_off, row_off, w, h):
    col_off = int(min(max(0, col_off), max(0, src.width  - 1)))
    row_off = int(min(max(0, row_off), max(0, src.height - 1)))
    w = int(min(w, src.width  - col_off))
    h = int(min(h, src.height - row_off))
    win = Window(col_off, row_off, w, h)
    arr = src.read(window=win, masked=True)
    # apply scales/offsets if present
    if src.scales and all(s is not None for s in src.scales):
        for i, scale in enumerate(src.scales, start=1):
            if scale not in (None, 1):
                arr[i-1] = arr[i-1] * scale
    if src.offsets and any(o not in (None, 0) for o in src.offsets):
        for i, off in enumerate(src.offsets, start=1):
            if off not in (None, 0):
                arr[i-1] = arr[i-1] + off
    return arr, win

def reproject_mask_to_roi(src_mask, dst_shape_hw, dst_transform, dst_crs):
    """ Vrati masku poravnatu na dati ROI grid (h,w). """
    h, w = dst_shape_hw
    dst = np.zeros((h, w), dtype=np.uint8)
    reproject(
        source=rasterio.band(src_mask, 1),
        destination=dst,
        src_transform=src_mask.transform,
        src_crs=src_mask.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=0,
        resampling=Resampling.nearest
    )
    return (dst > 0).astype(np.uint8)

def find_roi_inside_mask(src_img, src_mask, w, h, max_tries, min_mask_cov, min_valid_ratio):
    """ Traži ROI koji ima mask_coverage >= min_mask_cov i dovoljno signala (varijansa/valid pikseli). """
    # 1) pokušaj centralni prozor
    candidates = [
        ((src_img.width - w)//2, (src_img.height - h)//2),
    ]
    # 2) dodaj nekoliko determinističkih probnih tačaka (četiri kvadranta)
    candidates += [
        (0, 0),
        (max(0, src_img.width  - w), 0),
        (0, max(0, src_img.height - h)),
        (max(0, src_img.width  - w), max(0, src_img.height - h)),
    ]
    # 3) nasumično
    for _ in range(max_tries):
        candidates.append((
            random.randint(0, max(0, src_img.width  - w)),
            random.randint(0, max(0, src_img.height - h))
        ))

    tried = 0
    for col_off, row_off in candidates:
        tried += 1
        arr_img, win = read_window_masked(src_img, col_off, row_off, w, h)
        roi_transform = window_transform(win, src_img.transform)

        # maska poravnata na ovaj ROI
        mask_bin = reproject_mask_to_roi(src_mask, (int(win.height), int(win.width)), roi_transform, src_img.crs)
        mask_cov = mask_bin.sum() / (win.width * win.height + 1e-9)

        # signal u slici
        valid_ratio = (~arr_img.mask[0]).sum() / (win.width * win.height + 1e-9)
        gray = robust_gray_from_bands(arr_img)
        var_gray = float(np.nanvar(gray))

        if mask_cov >= min_mask_cov and valid_ratio >= min_valid_ratio and var_gray > 0:
            print(f"[INFO] Nađen ROI unutar maske na pokušaju {tried}: col_off={int(win.col_off)}, row_off={int(win.row_off)}")
            return arr_img, win, mask_bin

    print("[WARN] Nije pronađen ROI sa dovoljnom pokrivenošću maske — vraćam najbolji koji imamo (možda < traženog praga).")
    # fallback: uzmi centralni prozor sa pripadajućom maskom
    arr_img, win = read_window_masked(src_img, (src_img.width - w)//2, (src_img.height - h)//2, w, h)
    roi_transform = window_transform(win, src_img.transform)
    mask_bin = reproject_mask_to_roi(src_mask, (int(win.height), int(win.width)), roi_transform, src_img.crs)
    return arr_img, win, mask_bin

# ---- Glavni tok ----
with rasterio.open(image_input_path) as src_img, rasterio.open(mask_input_path) as src_mask:
    print(f"[INFO] Slika: {src_img.width}x{src_img.height}px, bands={src_img.count}, dtype={src_img.dtypes}, nodata={src_img.nodata}")
    # prikaži granice u istom CRS-u (prebaci masku u CRS slike)
    img_bounds = src_img.bounds
    mask_bounds_imgcrs = transform_bounds(src_mask.crs, src_img.crs, *src_mask.bounds, densify_pts=21)
    print(f"[INFO] Bounds (slika, {src_img.crs}): {img_bounds}")
    print(f"[INFO] Bounds (maska→slika CRS):      {mask_bounds_imgcrs}")

    # pronađi ROI koji je UNUTAR MASKE
    arr_img, win, mask_bin = find_roi_inside_mask(
        src_img, src_mask,
        ROI_WIDTH, ROI_HEIGHT,
        MAX_TRIES, MIN_MASK_COVERAGE, MIN_VALID_PIXELS
    )

    roi_transform = window_transform(win, src_img.transform)
    img_profile = src_img.profile.copy()
    h, w = int(win.height), int(win.width)

    mask_cov = mask_bin.sum() / (h * w + 1e-9)
    valid_counts = [int((~arr_img.mask[i]).sum()) for i in range(min(src_img.count, 3))]
    print(f"[INFO] ROI: col_off={int(win.col_off)}, row_off={int(win.row_off)}, w={w}, h={h}")
    print(f"[DEBUG] valid pix po bandu (prve 3): {valid_counts}")
    print(f"[DEBUG] mask coverage u ROI: {mask_cov*100:.2f}%")

# ---- Canny + maskiranje ----
gray = robust_gray_from_bands(arr_img)
tmin, tmax = float(np.nanmin(gray)), float(np.nanmax(gray))
var_gray = float(np.nanvar(gray))
print(f"[DEBUG] gray min/max = {tmin:.6f} / {tmax:.6f}, var={var_gray:.6e}")

edges_bool = canny(gray, sigma=CANNY_SIGMA, low_threshold=CANNY_LOW, high_threshold=CANNY_HIGH)
num_edges_raw = int(edges_bool.sum())
edges_u8 = (edges_bool.astype(np.uint8) * 255)
print(f"[DEBUG] ivice pre maske = {num_edges_raw}")

edges_masked = (edges_u8 * mask_bin).astype(np.uint8)
num_edges_masked = int((edges_masked > 0).sum())
print(f"[DEBUG] ivice posle maske = {num_edges_masked}")

if mask_cov < MIN_MASK_COVERAGE:
    print(f"[WARN] ROI mask coverage ({mask_cov*100:.2f}%) je ispod ciljanog praga ({MIN_MASK_COVERAGE*100:.0f}%). "
          "Možda su maska i slika samo delimično preklopljene — proveri bounds iznad.")

# ---- Snimi izlaze ----
out_profile = img_profile.copy()
out_profile.update({
    "width": w,
    "height": h,
    "transform": roi_transform,
    "count": 1,
    "dtype": "uint8",
    "nodata": 0
})
with rasterio.open(output_tif, "w", **out_profile) as dst:
    dst.write(edges_masked, 1)

imsave(output_png, edges_masked, check_contrast=False)

print(f"[OK] Snimljeno GeoTIFF: {output_tif}")
print(f"[OK] Snimljeno PNG:     {output_png}")
