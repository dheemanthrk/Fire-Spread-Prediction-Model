import matplotlib.pyplot as plt
import rasterio
stacked_tiff = "Fire_Temp_MultiBand.tif"
# Load Multi-Band GeoTIFF
with rasterio.open(stacked_tiff) as src:
    fire_mask = src.read(1)  # Band 1: Fire Mask
    temperature = src.read(2)  # Band 2: Temperature

# Plot Fire Mask
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(fire_mask, cmap="gray")
plt.colorbar(label="Fire (0=No Fire, 1=Fire)")
plt.title("🔥 Fire Mask")

# Plot Temperature
plt.subplot(1, 2, 2)
plt.imshow(temperature, cmap="hot")
plt.colorbar(label="Temperature (Kelvin)")
plt.title("🌡️ Temperature")

plt.show()
