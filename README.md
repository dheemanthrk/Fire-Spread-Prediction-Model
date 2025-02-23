# Fire-Spread-Prediction-Model
CNN + LSTM
# Forest Fire Spread Prediction

## Overview
This project focuses on collecting, processing, and analyzing meteorological and environmental data to support next-day fire spread prediction in Canada. The workflow integrates multiple data sources, including remote sensing and climate datasets, and processes them for machine learning applications.

## Data Sources
### 1️⃣ **Meteorological Data**
- **ERA5-Land (ECMWF/ERA5_LAND)** - Provides daily average temperature, wind speed, and other climate variables.
- **MSC GeoMet API (Environment Canada)** - Official Canadian weather data.

### 2️⃣ **Vegetation Data**
- **MODIS (`MODIS/006/MOD11A1`)** - Daily land surface temperature.
- **Copernicus Sentinel-3 SLSTR** - High-resolution land surface temperature.

### 3️⃣ **Historical Fire Data**
- **CWFIS API (Canadian Wildland Fire Information System)** - Provides fire history records for Canada.

### 4️⃣ **Topographical Data**
- **ABMPS Province of BC Shapefile** - Used to create a **10 km x 10 km grid** for spatial mapping.
- **Downloaded 40 TIFF files** for topographical data from British Columbia.

## Data Processing Workflow
### ✅ **1. Data Collection**
- Fetch meteorological, NDVI, and fire history data using APIs.
- Extract essential climate variables like temperature, humidity, and wind speed.

### ✅ **2. Data Preprocessing**
- Convert day numbers to dates for alignment across datasets.
- Filter and merge datasets with a **10 km x 10 km grid**.
- Handle missing values and consider interpolation if needed.

### ✅ **3. Multi-Channel GeoTIFF for CNN + LSTM**
- **We are creating a multi-channel GeoTIFF file as input for a CNN + LSTM model.**
- **Initially, we are using only temperature and fire mask layers as inputs before expanding to additional features.**
- **This allows us to validate the approach before incorporating more environmental variables.**

### ✅ **4. Export Formats**
- **Multi-Channel GeoTIFF (`.tif`)** for CNN + LSTM model input.
- **CSV (`.csv`)** for tabular data storage and ML training.
- **PNG/JPG (`.png`, `.jpg`)** for quick visualizations.

## Sample CNN + LSTM Model Code
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Reshape

# Define CNN + LSTM Model
model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=(7, 64, 64, 2)),
    TimeDistributed(MaxPooling2D(pool_size=(2, 2))),
    TimeDistributed(Flatten()),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Predicting fire spread probability for next day
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

## Next Steps
- **Expand the multi-channel GeoTIFF to include NDVI, topographical, and fire history data.**
- **Train CNN + LSTM models using the multi-channel dataset.**
- **Evaluate model accuracy and optimize hyperparameters.**

## Acknowledgments
- Google Earth Engine, NASA, Copernicus, and Environment Canada for providing open-access climate and fire data.

---
🚀 **Stay tuned for more updates!**

