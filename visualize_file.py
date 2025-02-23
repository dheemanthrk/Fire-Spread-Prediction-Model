import geopandas as gpd
import matplotlib.pyplot as plt

# Load Fire Data
fire_shapefile = "/Users/dheemanth/Desktop/Cistel/MODIS Active Fire Data/MODIS_C6_1_Canada_7d/MODIS_C6_1_Canada_7d.shp"
fire_data = gpd.read_file(fire_shapefile)

# Plot fire points
fig, ax = plt.subplots(figsize=(8, 6))
fire_data.plot(ax=ax, color='red', markersize=10, label="Fire Points")
ax.set_title("🔥 Fire Locations (MODIS)")
plt.legend()
plt.show()
