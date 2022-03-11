# Steps to have everything up and running

# PRE-REQUISITES

# Create a virtual environment
virtualenv venv 

# Activate the Virtual Environmnet
source venv/bin/activate

# Install all the requirements needed for the project
pip install -r requirements.txt


# EXPERIMENTATION

# Setting up the credentials of AWS and storing them in a ".env" file
python programs/00_set_credentials.py

# Download the geolocated data from S3 (geolife_consolidated.parquet)
python programs/01.2_s3_download_data.py

# Split the geolocated data into files, a file per user
# Note:     -cw, -co    are put to True to clean Weekends and Outliers 
python programs/01.3_geolife_users_split.py -cw True -co True

# Execute the GAN training and the storage of results
python programs/02.6_gan_generation_vTerminal.py

# Load the report of all the evaluations to S3
python programs/02.7_gan_reporting.py