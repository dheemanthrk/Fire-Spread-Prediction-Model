# Fire-Spread-Prediction-Model

## Overview
This project focuses on collecting, processing, and analyzing meteorological and environmental data to support next-day fire spread prediction in Canada. The workflow integrates multiple data sources, including remote sensing and climate datasets, and processes them for machine learning applications.

## Data Sources
### 1️⃣ **Meteorological Data**
- **ERA5-Land (ECMWF/ERA5_LAND)** - Provides daily average temperature, wind speed, and other climate variables.
- **MSC GeoMet API (Environment Canada)** - Official Canadian weather data.

## Data Processing Workflow
### ✅ **1. Data Collection**
- Fetch temperature data using APIs and google earth engine.
- Extract essential climate variables like temperature.

### ✅ **2. Data Preprocessing**
- Convert day numbers to dates for alignment across datasets.

### ✅ **3. Multi-Channel GeoTIFF for CNN + LSTM**
- **We are creating a multi-channel GeoTIFF file as input for a CNN + LSTM model.**
- **Initially, we are using only temperature and fire mask layers as inputs before expanding to additional features.**
- **This allows us to validate the approach before incorporating more environmental variables.**

### ✅ **4. Export Formats**
- **Multi-Channel GeoTIFF (`.tif`)** for CNN + LSTM model input.
- **CSV (`.csv`)** for tabular data storage and ML training.
- **PNG/JPG (`.png`, `.jpg`)** for quick visualizations.

## Sample CNN + LSTM Model Code
### 🔹 **Function to Load Multi-Channel TIFF Files**
```python
import rasterio
import numpy as np

def load_multichannel_tiff(filepath):
    with rasterio.open(filepath) as src:
        data = src.read()  # Shape: (channels, height, width)
    return np.moveaxis(data, 0, -1)  # Move channels to the last dimension

# Example Usage
tiff_file = "path/to/multichannel.tif"
image_data = load_multichannel_tiff(tiff_file)  # Shape: (height, width, channels)
image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
```

### 🔹 **Building the CNN + LSTM Model**
```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Reshape, Input

# Define Model Input Shape: (time_steps, height, width, channels)
time_steps = 7
height, width, channels = image_data.shape[1], image_data.shape[2], image_data.shape[3]
input_shape = (time_steps, height, width, channels)

# Build CNN + LSTM Model
input_layer = Input(shape=input_shape)

# CNN Feature Extraction (applied to each time step)
x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(input_layer)
x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
x = TimeDistributed(Flatten())(x)

# LSTM for Temporal Dependencies
x = LSTM(64, return_sequences=False)(x)

# Fully Connected Layers
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)  # Predicting fire spread probability

# Compile Model
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print Model Summary
model.summary()
```

### 🔹 **Example Prediction Using Dummy Data**
```python
# Predict Next-Day Fire Spread (Example)
prediction = model.predict(np.random.rand(1, time_steps, height, width, channels))  # Dummy input
print(f"Fire Spread Probability: {prediction[0][0]:.4f}")
```

## Next Steps
- **Expand the multi-channel GeoTIFF to include NDVI, topographical, and fire history data.**
- **Train CNN + LSTM models using the multi-channel dataset.**
- **Evaluate model accuracy and optimize hyperparameters.**

## Acknowledgments
- Google Earth Engine, NASA, Copernicus, and Environment Canada for providing open-access climate and fire data.

---
🚀 **Stay tuned for more updates!**


