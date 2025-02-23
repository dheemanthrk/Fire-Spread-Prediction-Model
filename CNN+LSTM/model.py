import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Reshape, Input
import rasterio
import numpy as np

# Function to load a multi-channel GeoTIFF file
def load_multichannel_tiff(filepath):
    with rasterio.open(filepath) as src:
        data = src.read()  # Shape: (channels, height, width)
    return np.moveaxis(data, 0, -1)  # Move channels to the last dimension

# Example Usage
tiff_file = "path/to/multichannel.tif"
image_data = load_multichannel_tiff(tiff_file)  # Shape: (height, width, channels)
image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

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
# x = LSTM(64, return_sequences=False)(x)

# # Fully Connected Layers
# x = Dense(64, activation='relu')(x)
# output = Dense(1, activation='sigmoid')(x)  # Predicting fire spread probability

# # Compile Model
# model = Model(inputs=input_layer, outputs=output)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Print Model Summary
# model.summary()

# # Predict Next-Day Fire Spread (Example)
# prediction = model.predict(np.random.rand(1, time_steps, height, width, channels))  # Dummy input
# print(f"Fire Spread Probability: {prediction[0][0]:.4f}")
