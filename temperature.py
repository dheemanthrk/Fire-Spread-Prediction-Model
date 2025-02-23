import ee
import geemap


ee.Initialize()


region = [
    [-130, 30], [-60, 30], [-60, 50], [-130, 50], [-130, 30]
]


start_date = "2024-02-01"
end_date = "2024-02-07"


dataset = ee.ImageCollection("IDAHO_EPSCOR/GRIDMET") \
    .filterDate(start_date, end_date) \
    .select("tmmx") 


temp_image = dataset.mean().clip(ee.Geometry.Polygon([region]))

task = ee.batch.Export.image.toDrive(
    image=temp_image,
    description="Temperature_FireRegion",
    folder="EarthEngine",
    fileNamePrefix="Temperature_2024_02_01_07",
    region=region, 
    scale=4000, 
    crs="EPSG:4326",
    fileFormat="GeoTIFF"
)
task.start()

print("✅ Temperature Data Download Started! Check Google Drive in a few minutes.")
