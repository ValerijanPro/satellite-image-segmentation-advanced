

# Ovaj kod od ulazne TIF satelitske slike pravi izlazni TIF fajl segmentacione maske (final_sam_mask_planet.tif unutar "output" foldera)

import os
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from rasterio.features import shapes
from samgeo import SamGeo
import torch

#print("torch.cuda.is_available():", torch.cuda.is_available())

# Input files

image_input_path = "images/composite.tif"
mask_input_path = "ParcelaMaska.tif"
sam_checkpoint_path = "sam_vit_l_0b3195.pth"

# Tiling parameters
tile_size = 1536
tile_overlap = 192
tile_stride = tile_size - tile_overlap
SAM_batch_tuple = (1024, 1024)
SAM_kernel = (0, 0)
max_tiles = 10 # 9 min per tile
skip_first_n_tiles = 15

# Output paths
final_sam_mask_tif = "output/final_sam_mask_planet.tif"
tiles_folder = "output/tiles_sam_planet"

##############################################
# 2) READ THE FULL IMAGE
##############################################
with rasterio.open(image_input_path) as src_image:
    image_full_width = src_image.width
    image_full_height = src_image.height
    image_transform = src_image.transform
    image_crs = src_image.crs
    image_profile = src_image.profile.copy()

    image_data_full = src_image.read()  # (bands, height, width)


##############################################
# 3) READ & REPROJECT/RESAMPLE FARMLAND MASK
##############################################
with rasterio.open(mask_input_path) as src_mask:
    mask_data = src_mask.read(1)  # farmland mask (single band)
    mask_transform = src_mask.transform
    mask_crs = src_mask.crs

    # Reproject if needed
    if mask_crs != image_crs:
        print("Reprojecting farmland mask to match image CRS...")
        mask_data_aligned = np.zeros(
            (image_full_height, image_full_width),
            dtype=mask_data.dtype
        )
        reproject(
            source=mask_data,
            destination=mask_data_aligned,
            src_transform=mask_transform,
            src_crs=mask_crs,
            dst_transform=image_transform,
            dst_crs=image_crs,
            resampling=Resampling.nearest
        )
        mask_data = mask_data_aligned
    else:
        # If CRS matches but shape differs, resample
        if mask_data.shape != (image_full_height, image_full_width):
            print("Resampling farmland mask to match image shape...")
            mask_data_aligned = np.zeros(
                (image_full_height, image_full_width),
                dtype=mask_data.dtype
            )
            reproject(
                source=mask_data,
                destination=mask_data_aligned,
                src_transform=mask_transform,
                src_crs=mask_crs,
                dst_transform=image_transform,
                dst_crs=image_crs,
                resampling=Resampling.nearest
            )
            mask_data = mask_data_aligned

##############################################
# 4) APPLY THE FARMLAND MASK (0 OR 1) TO IMAGE
##############################################
mask_bin = (mask_data > 0).astype(np.uint8)
# Multiply across all bands => zero out non-farmland
masked_image_full = image_data_full * mask_bin

##############################################
# 5) PREPARE A SAM MODEL
##############################################
sam = SamGeo(
    model_type="vit_l",
    checkpoint=sam_checkpoint_path,
    device="cpu"  # "cuda"
)

##############################################
# 6) SETUP A FINAL SAM MASK ARRAY
##############################################
# We'll store 0/255 values across the entire image
final_sam_mask = np.zeros((image_full_height, image_full_width), dtype=np.uint8)

# Create a folder to store intermediate tile outputs
os.makedirs(tiles_folder, exist_ok=True)

##############################################
# 7) TILE LOOP WITH OPTIONAL 16->8 bit CONVERSION
##############################################
tile_count = 0
row = 0

while row < image_full_height:
    col = 0
    row_end = min(row + tile_size, image_full_height)

    while col < image_full_width:
        col_end = min(col + tile_size, image_full_width)

        # 1) Extract the tile for all bands => shape (bands, tile_height, tile_width)
        tile_data_16 = masked_image_full[
            :,
            row:row_end,
            col:col_end
        ]

        # 2) Check if tile is completely black
        if (np.any(tile_data_16 != 0) and skip_first_n_tiles == 0):
            # 2a) Convert tile_data to 3-band, 8-bit
            #     i.e., pick up to first 3 bands and rescale to 0..255.

            bands_count = tile_data_16.shape[0]
            max_bands = min(bands_count, 3)  # keep 1..3 channels
            tile_data_16 = tile_data_16[:max_bands, :, :]  # shape => (3 or fewer, h, w)

            # Convert to float for scaling
            tile_data_float = tile_data_16.astype(np.float32)

            # Simple min-max stretch
            min_val = tile_data_float.min()
            max_val = tile_data_float.max()
            if max_val > min_val:
                tile_data_float = (tile_data_float - min_val) / (max_val - min_val) * 255.0
            else:
                # tile is constant => set all to 0
                tile_data_float[:] = 0.0

            tile_data_8 = tile_data_float.astype(np.uint8)  # now shape=(channels, h, w), dtype=uint8

            # In-memory tile transform
            tile_window = Window(
                col_off=col,
                row_off=row,
                width=(col_end - col),
                height=(row_end - row)
            )
            tile_transform = window_transform(tile_window, image_transform)

            # 2b) Prepare tile profile for 3-band, 8-bit
            tile_profile = image_profile.copy()
            tile_profile.update({
                "width": (col_end - col),
                "height": (row_end - row),
                "transform": tile_transform,
                "count": max_bands,     # Number of bands
                "dtype": "uint8"
            })

            # 2c) Write the 8-bit tile to disk
            tile_input_path = os.path.join(
                tiles_folder,
                f"tile_r{row}_c{col}_input.tif"
            )
            with rasterio.open(tile_input_path, "w", **tile_profile) as dst:
                dst.write(tile_data_8)

            # 3) Run SAM on this tile
            tile_output_path = os.path.join(
                tiles_folder,
                f"tile_r{row}_c{col}_sam_mask.tif"
            )
            sam.generate(
                source=tile_input_path,
                output=tile_output_path,
                batch=True,
                batch_sample_size=SAM_batch_tuple,
                foreground=False,
                erosion_kernel=SAM_kernel,
                mask_multiplier=255
            )

            # 4) Read the tile-level SAM mask & threshold
            with rasterio.open(tile_output_path) as src_sam_tile:
                tile_sam_arr = src_sam_tile.read(1)
            tile_sam_bin = np.where(tile_sam_arr >= 128, 255, 0).astype(np.uint8)

            # 5) Combine with final SAM mask (via max)
            final_sam_mask[row:row_end, col:col_end] = np.maximum(
                final_sam_mask[row:row_end, col:col_end],
                tile_sam_bin
            )

            tile_count += 1
            if max_tiles is not None and tile_count >= max_tiles:
                print("Reached max_tiles limit; stopping early.")
                break

        elif skip_first_n_tiles != 0:
            skip_first_n_tiles -= 1
        # Move to next column
        col += tile_stride
        if max_tiles is not None and tile_count >= max_tiles:
            break

    # Move to next row
    row += tile_stride
    if max_tiles is not None and tile_count >= max_tiles:
        break



##############################################
# 8) SAVE THE FINAL SAM MASK (ENTIRE IMAGE)
##############################################
# We use the same profile as the original image, but single band
final_profile = image_profile.copy()
final_profile.update({
    "count": 1,
    "dtype": str(final_sam_mask.dtype)
})

with rasterio.open(final_sam_mask_tif, "w", **final_profile) as dst:
    dst.write(final_sam_mask, 1)

print("Final SAM mask (assembled from all tiles) saved to:", final_sam_mask_tif)