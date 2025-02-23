# Forest Fire Spread Prediction (64x64 Grid)

## Overview
This project focuses on collecting, processing, and analyzing meteorological and environmental data to support next-day fire spread prediction in Canada. The workflow integrates multiple data sources, including remote sensing and climate datasets, and processes them into a **64 km x 64 km grid (1 km cell size)** for machine learning applications.

## Data Sources
### 1️⃣ **MODIS Active Fire Data**
- Contains fire event data, including latitude, longitude, brightness, and fire confidence levels.
- Used to create a **fire mask raster** where pixels represent fire occurrences (1 = Fire, 0 = No Fire).

### 2️⃣ **ERA5-Land Temperature Data**
- Provided by ECMWF through Google Earth Engine.
- Captures daily **surface temperature (Kelvin)** over the selected fire-prone area.

## Data Processing Workflow
### ✅ **1. Select Fire-Prone Region in Canada**
- Focus on **British Columbia (Cariboo Region)**, a high-risk wildfire area.
- Use a **64 km x 64 km bounding box**, centered on MODIS fire events.

### ✅ **2. Convert MODIS Fire Data to a 64x64 Grid**
- **Create a raster grid** where each cell is **1 km x 1 km**.
- **Rasterize fire events** into a binary mask (1 = Fire, 0 = No Fire).

#### 🔹 **Python Code: Convert MODIS Fire Data to Raster**
```python
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import Point

# 🔥 Load MODIS Fire Data (from CSV)
fire_csv = "MODIS_Active_Fires.csv"
fire_df = pd.read_csv(fire_csv)

# Convert DataFrame to GeoDataFrame
fire_gdf = gpd.GeoDataFrame(
    fire_df,
    geometry=gpd.points_from_xy(fire_df.longitude, fire_df.latitude),
    crs="EPSG:4326"
)

# 🌎 Define Raster Grid for 64 km x 64 km Area
resolution = 0.01  # 1 km cell size
grid_size = 64
minx, miny, maxx, maxy = fire_gdf.total_bounds
center_x, center_y = (minx + maxx) / 2, (miny + maxy) / 2

# Bounding box for 64x64 grid
minx, miny = center_x - (grid_size / 2) * resolution, center_y - (grid_size / 2) * resolution
maxx, maxy = center_x + (grid_size / 2) * resolution, center_y + (grid_size / 2) * resolution

# 🖼️ Rasterize Fire Locations
fire_raster = rasterize(
    [(geom, 1) for geom in fire_gdf.geometry],
    out_shape=(grid_size, grid_size),
    transform=rasterio.transform.from_origin(minx, maxy, resolution, resolution),
    fill=0,
    dtype=np.uint8
)

# 💾 Save Fire Raster as GeoTIFF
fire_tiff = "MODIS_Fire_Mask_64x64.tif"
with rasterio.open(fire_tiff, "w", driver="GTiff", height=grid_size, width=grid_size, count=1, dtype=np.uint8, crs="EPSG:4326", transform=rasterio.transform.from_origin(minx, maxy, resolution, resolution)) as dst:
    dst.write(fire_raster, 1)

print(f"✅ Fire Mask GeoTIFF Created: {fire_tiff}")
```

### ✅ **3. Fetch ERA5 Temperature for the 64 km x 64 km Region**
- **Retrieve temperature data** for the same grid.
- **Clip to match the fire mask grid.**

#### 🔹 **Python Code: Fetch ERA5 Temperature for 64x64 Grid**
```python
import ee

# Initialize Google Earth Engine
ee.Initialize()

# Define Region (64 km x 64 km, centered on fire locations)
region = ee.Geometry.Rectangle([minx, miny, maxx, maxy])

# Define Date Range
start_date = "2025-02-01"
end_date = "2025-02-07"

# Load ERA5-Land Temperature Data
dataset = ee.ImageCollection("ECMWF/ERA5_LAND") \n    .filterDate(start_date, end_date) \n    .select("temperature_2m")

temp_image = dataset.mean().clip(region)

task = ee.batch.Export.image.toDrive(
    image=temp_image,
    description="ERA5_Temperature_64x64",
    folder="EarthEngine",
    fileNamePrefix="ERA5_Temperature_2025_02_01_07",
    region=region,
    scale=1000,
    crs="EPSG:4326",
    fileFormat="GeoTIFF"
)
task.start()

print("✅ ERA5 Temperature Data Download Started! Check Google Drive in a few minutes.")
```

### ✅ **4. Merge Fire Mask & Temperature Data into Multi-Channel GeoTIFF**
- **Band 1:** Fire Mask
- **Band 2:** ERA5 Temperature

#### 🔹 **Python Code: Create Multi-Channel TIFF**
```python
import rasterio
import numpy as np

# 🔥 Load Fire Mask
fire_tiff = "MODIS_Fire_Mask_64x64.tif"
with rasterio.open(fire_tiff) as fire_src:
    fire_data = fire_src.read(1)
    fire_meta = fire_src.meta

# 🌡️ Load ERA5 Temperature Data
temp_tiff = "ERA5_Temperature_2025_02_01_07.tif"
with rasterio.open(temp_tiff) as temp_src:
    temp_data = temp_src.read(1)
    temp_meta = temp_src.meta

# Ensure dimensions match
if fire_data.shape != temp_data.shape:
    raise ValueError("Fire mask and temperature data dimensions do not match!")

# 📂 Create Multi-Channel GeoTIFF
multi_tiff = "MultiChannel_Fire_Temperature_64x64.tif"
multi_meta = fire_meta.copy()
multi_meta.update(count=2)

with rasterio.open(multi_tiff, "w", **multi_meta) as dst:
    dst.write(fire_data, 1)
    dst.write(temp_data, 2)

print(f"✅ Multi-Channel GeoTIFF Created: {multi_tiff}")
```

## Next Steps

### ✅ **5. Visualize the Multi-Channel TIFF**
- Load and display the **Fire Mask** and **Temperature** bands.
- Use Matplotlib to inspect raster layers.

#### 🔹 **Python Code: Visualizing Multi-Channel TIFF**
```python
import rasterio
import matplotlib.pyplot as plt

# Load Multi-Channel GeoTIFF
with rasterio.open("MultiChannel_Fire_Temperature_64x64.tif") as src:
    fire_mask = src.read(1)
    temperature = src.read(2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Fire Mask")
plt.imshow(fire_mask, cmap="Reds")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Temperature (K)")
plt.imshow(temperature, cmap="coolwarm")
plt.colorbar()

plt.show()
```

### ✅ **6. Train CNN + LSTM Model with Multi-Channel Data**
- Implement a **CNN to extract spatial features**.
- Integrate an **LSTM layer for temporal dependencies**.
- Use the **Multi-Channel GeoTIFF** as input to predict fire spread.

#### 🔹 **Python Code: CNN + LSTM Model Architecture**
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Reshape, Input

# Define CNN + LSTM Model
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(7, 64, 64, 2)),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Predicting fire spread probability
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

🚀 **Stay tuned for more updates!**
1️⃣ **Visualize the Multi-Channel TIFF**
2️⃣ **Train CNN + LSTM with Multi-Channel Data**

🚀 **Stay tuned for more updates!**

