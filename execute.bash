# Steps to have everything up and running

# Download the geolocated data from S3 (geolife_consolidated.parquet)
python programs/01.2_s3_download_data.py

# Split the geolocated data into files, a file per user
# Note:     -cw, -co    are put to True to clean Weekends and Outliers 
python programs/01.3_geolife_users_split.py -cw True -co True
