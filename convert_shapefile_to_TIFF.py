import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

# 🔥 Load Fire Data (Shapefile)
fire_shapefile = "FIRMS_MODIS_fire.shp"  # Update with your file name
fire_data = gpd.read_file(fire_shapefile)

# 🔍 Check the CRS (Coordinate Reference System)
print("Original CRS:", fire_data.crs)

# If needed, reproject to EPSG:4326 (WGS84 - Lat/Lon)
if fire_data.crs is None or fire_data.crs != "EPSG:4326":
    fire_data = fire_data.to_crs("EPSG:4326")

# 🌎 Define Raster Grid (Resolution & Extent)
resolution = 0.01  # Approx. 1 km resolution
minx, miny, maxx, maxy = fire_data.total_bounds  # Bounding box
width = int((maxx - minx) / resolution)
height = int((maxy - miny) / resolution)

# 🖼️ Rasterize Fire Locations
fire_raster = rasterize(
    [(geom, 1) for geom in fire_data.geometry],  # Set fire pixels to 1
    out_shape=(height, width),
    transform=rasterio.transform.from_origin(minx, maxy, resolution, resolution),
    fill=0,  # Background = 0 (no fire)
    all_touched=True,
    dtype=np.uint8
)

# 💾 Save as GeoTIFF
output_tiff = "MODIS_Fire_Mask.tif"
with rasterio.open(
    output_tiff, "w",
    driver="GTiff",
    height=height,
    width=width,
    count=1,
    dtype=np.uint8,
    crs="EPSG:4326",
    transform=rasterio.transform.from_origin(minx, maxy, resolution, resolution)
) as dst:
    dst.write(fire_raster, 1)

print(f"✅ Fire Mask GeoTIFF Created: {output_tiff}")
