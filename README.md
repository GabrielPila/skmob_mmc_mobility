# DATA PRIVACY ON GEOLOCATED DATASETS

This repository storage a project of Data Privacy on Geolocated Datasets through synthetic data generation using Generative Adversarial Networks. 
The dataset used for the experiments is the [Geolife DataSet](https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F), which is available on the Microsoft Research website.

The project had primarily the following steps:
1. Extract and preprocess de geolocated dataset
2. Generate algorithms to estimate the Mobility Markok Chain (MMC) representations per each user.
3. Generate Syntethic mobility data with GANs to evaluate the privacy preserving capabilities of the experiment and the remaining usefulness of the data. 

You can find a more extensive documentation in the `./docs/` directory.

# 1. Installation and Set Up

## 1.1. Dependencies
1. Clone de repository: 
`git clone git@github.com:GabrielPila/skmob_mmc_mobility.git`
2. Access the repo: 
`cd skmob_mmc_mobility`
3. Create a virtual environment:
`virtualenv venv`
4. Install dependencies:
`pip install -r requirements.txt`
5. Activate your environmnet:
`source venv/bin/activate`
6. Now your enviroment is ready to work!

## 1.2. Data Gathering

### 1.2.1. Downloading and Processing Information (optional)

If you want to download the data from the Microsoft Research website, observe its original organization and to process the data in your device, run the following scripts:
> `python3 programs/geolife_extraction.py`

### 1.2.2. Downloading Processed Information

We already processed the data and uploaded to a public path in Google Drive. If you want to get the already processed dataset, you can download the information with the following scripts:
```bash
# Data extracted 
# after downloading Microsoft data and appending the information of the files
!gdown https://drive.google.com/uc?id=1gAJ5LXOWXPbzGLDYzVTYSKDPK7KThGgT

# Data consolidated [PREFERRED]
# this dataset contains an estimation of the distance between continuous points. 
!gdown https://drive.google.com/uc?id=1h9RohsM_Z9w-Ny_WHwZt856tljl5I39J

# Sample data [TO RUN SMALL EXPERIMENTS]
!gdown https://drive.google.com/uc?id=XXXXXXXXX
```

# 2. Mobility Markov Chains


# 3. Syntethic Data Generation

# Z. To DO
- Reorganize functions into `utils.py`
- Clean the MMC generation process
- Clean the House location process.


# Contributors:
- Gabriel Pila (gabriel.pilah@pucp.edu.pe)
- Anthony Ruiz (ruizc.anthony@gmail.com)
