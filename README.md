# eda
EDA and analysis on our dataset

# Focus
Analyzing and processing the dataset.

# Datasets on shared Google Drive
- The "loadable data" (filtered bad data) is located on the shared drive at https://drive.google.com/drive/folders/1WmdBnrZVIR_I69sMW7i-lcBW-qjQopJl?usp=drive_link.
- The pre-filter (i.e., contains bad data) is located on the shared drive at https://drive.google.com/drive/folders/1QNuhGxLAC3WWOeK8v7KwgA0oQbD2TuoX?usp=drive_link.

# Dataset from Google bucket
The primary dataset for this project ("loadable_Combined_HU_HT.tar.gz") is downloaded from Google Cloud Storage using the scripts provided in this repository. This ensures you get the correct version of the data integrated with the project setup.

The script will download the data to a `data_dump` directory in your project root.

# References
TypeNet paper:\
Acien, A., Morales, A., Monaco, J. V., Vera-Rodríguez, R., & Fiérrez, J. (2021). TypeNet: Deep Learning Keystroke Biometrics. IEEE Transactions on Biometrics, Behavior, and Identity Science. arXiv:2101.05570. https://doi.org/10.48550/arXiv.2101.05570 

# Setup

This guide walks you through setting up the complete research environment and data access.

## Quick Start

For team members, follow these steps:

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
source activate.sh
# If that doesn't work try this
# source ./venv-3.12.5/bin/activate
# On Windows:
# activate.bat

# 4. Download the research data
# On Linux/Mac:
./download_data.sh
# If you are connecting remotely, use the --headless flag
# ./download_data.sh --headless
# On Windows (if a download_data.bat is provided):
# download_data.bat


