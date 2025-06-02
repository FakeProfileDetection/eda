# eda
EDA and analysis on our dataset

# Focus
Analyzing and processing the dataset.

# Datasets
### Raw data from HU_HT
- The "loadable data" (filtered bad data) is located on the shared drive at https://drive.google.com/drive/folders/1WmdBnrZVIR_I69sMW7i-lcBW-qjQopJl?usp=drive_link.
- The pre-filter (i.e., contains bad data) is located on the shared drive at https://drive.google.com/drive/folders/1QNuhGxLAC3WWOeK8v7KwgA0oQbD2TuoX?usp=drive_link.
- (Optional) `download_data.sh` will download the data to a `data_dump` directory in your project root.

### Processed data (key1-key2 press pairing)
- use `./download_processed_data.sh`

### Machine learning features data
- `ml-experients-with-outliers2025-05-31_142307`
- `ml-experients-without-outliers2025-05-31_143027`
- Also, both are stored in the shared drive: `https://drive.google.com/drive/folders/1d7VEy-tj9SRFstBrOXYus95j2H9qraeO?usp=drive_link`





# References
TypeNet paper:\
Acien, A., Morales, A., Monaco, J. V., Vera-Rodríguez, R., & Fiérrez, J. (2021). TypeNet: Deep Learning Keystroke Biometrics. IEEE Transactions on Biometrics, Behavior, and Identity Science. arXiv:2101.05570. https://doi.org/10.48550/arXiv.2101.05570 

Outliers:\
G Ismail M, Salem MA, Abd El Ghany MA, Aldakheel EA, Abbas S. Outlier detection for keystroke biometric user authentication. PeerJ Comput Sci. 2024 Jun 17;10:e2086. doi: 10.7717/peerj-cs.2086. PMID: 38983219; PMCID: PMC11232596.  https://pmc.ncbi.nlm.nih.gov/articles/PMC11232596/ 

# Setup

This guide walks you through setting up the complete research environment and data access.

## Set up your environment:

```bash
# 1. Clone the repository
git clone https://github.com/FakeProfileDetection/eda.git
# If you have ssh setup, use this
# git clone git@github.com:FakeProfileDetection/eda.git
cd  eda

# 2. Make the setup and data download scripts executable (Linux/Mac only)
chmod +x setup.sh
chmod +x download_data.sh
chmod +x utils.sh # (Utility script, also needs to be executable if directly run, though usually sourced)

# 3. Set up the Python environment
# On Linux/Mac:
./setup.sh
# On Windows (if a setup.bat is provided and updated for this flow):
# setup.bat # Ensure setup.bat is updated for the new two-step process

# 4. Activate the environment
source ./venv-3.12.5/bin/activate

# On Windows:
# activate.bat
```
# Use extracted features
This is the best optionn if someone has already processed the datasets. 
- `ml-experients-with-outliers2025-05-31_142307`
- `ml-experients-without-outliers2025-05-31_143027`

## Alternatively, you can re-process the data and then extract the features
You'll need to do this for new datasets.

#### Extract ke1-key2 paired typenet features: 
```bash
# 1. Download the research data
# On Linux/Mac:
./download_data.sh
# If you are connecting remotely, use the --headless flag
# ./download_data.sh --headless
# On Windows (if a download_data.bat is provided):
# download_data.bat

# Extract key1-key2 pairing and typenet features
python typenet_ml_extraction_polars.py
```
#### Upload newly extracted features to the cloud
You can upload your processed dataset to the cloud to make it available to other team members.  However, the default behavior when pulling a processed dataset is to use the most recent based on timestamp.  Only upload if you want everyone to use your dataset.  (Although, users can choose to dowload all processed dataset or select processed datasets...but they need to know which one.)

```bash
# To upload the dataset and making availble to other team members
#./upload_processed_data.sh processed_data-<timestamp>-<hostname>

# If you already have an archive, skip packing:
# ./upload_processed_data.sh --no-archive processed_data-20250521T143200-myhost.tar.gz

```
#### Download extracted features
```bash
# Default (latest):
./download_processed_data.sh

# Fetch *all* snapshots:
# ./download_processed_data.sh --all

# Fetch only those from a given host:
#./download_processed_data.sh --hostname myhost

# Pick interactively from the list:
#./download_processed_data.sh --interactive

```

#### Extract the ml features for training
```bash
python typenet_ml_features_polars.py
```





