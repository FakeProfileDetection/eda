# eda
EDA and analysis on our dataset

# Foucs
Analyzing and processing the dataset.

# Datasets
- The "loadable data" (filtered bad data) is located on the shared drive at https://drive.google.com/drive/folders/1WmdBnrZVIR_I69sMW7i-lcBW-qjQopJl?usp=drive_link.
- The pre-filter (i.e., contains bad data) is located on the shared drive at https://drive.google.com/drive/folders/1QNuhGxLAC3WWOeK8v7KwgA0oQbD2TuoX?usp=drive_link.

# References
TypeNet paper:\
Acien, A., Morales, A., Monaco, J. V., Vera-Rodríguez, R., & Fiérrez, J. (2021). TypeNet: Deep Learning Keystroke Biometrics. IEEE Transactions on Biometrics, Behavior, and Identity Science. arXiv:2101.05570. https://doi.org/10.48550/arXiv.2101.05570 

# Setup

This guide walks you through setting up the complete research environment and data access.

## Quick Start

For team members in a hurry, run these commands:

```bash
# Clone the repository
git clone https://github.com/your-team/your-repo.git
cd your-repo

# Make the setup script executable (Linux/Mac only)
chmod +x setup.sh

# Setup everything (environment + data)
# On Linux/Mac:
./setup.sh
# On Windows:
setup.bat

# Activate the environment
# On Linux/Mac:
source activate.sh
# On Windows:
activate.bat

# Start Jupyter Lab
jupyter-lab
```

## What Gets Set Up

The setup script creates a complete environment with:

1. **Python 3.12.5** virtual environment (consistent for all team members)
2. **GPU-optimized PyTorch** (if a compatible GPU is detected)
3. **Jupyter Lab** with a dedicated kernel for this project
4. **Research data** downloaded from our shared Google Drive
5. **Core data science packages**:
   - polars, pandas, numpy, scikit-learn
   - matplotlib, seaborn
   - jupyterlab, ipykernel
   - and more

## Python Version Options

The setup script provides several options if Python 3.12.5 is not found:

1. **Install Python 3.12.5 locally** in the project directory
2. **Install using pyenv** if available on your system
3. **Install manually** from python.org or your system's package manager

## For pyenv Users

The setup script has improved pyenv integration:

- Properly initializes pyenv in the current shell
- Detects Python 3.12.5 in your pyenv installations
- Offers to install it via pyenv if not found
- Creates a properly isolated virtual environment

## Using the Environment

### Activating the Environment

```bash
# On Linux/Mac:
source activate.sh

# On Windows:
activate.bat
```

### Starting Jupyter Lab

After activating the environment:

```bash
jupyter-lab
```

### Working with the Research Data

After setup, the data is available in the `data_dump` directory and via the `DATA_PATH` environment variable:

```python
import os
import pandas as pd

# Get the data directory
data_path = os.environ['DATA_PATH']

# Load a specific data file
df = pd.read_csv(f"{data_path}/user-1/data1.csv")
```

### Using the Helper Module

We provide a helper module for common data operations:

```python
from team_data import data

# Get all available user IDs
user_ids = data.get_user_ids()

# Load data for a specific user
user_data = data.load_user_file(1, "data1.csv")
```

## Troubleshooting

### Python Version Issues

If you encounter problems with Python 3.12.5:

- **For pyenv users**: Run `pyenv init` and follow the instructions to properly set up your shell
- **For local Python**: If the local Python build fails, install build dependencies first:
  - Ubuntu/Debian: `sudo apt-get install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev`
  - macOS with Homebrew: `brew install openssl readline sqlite3 xz zlib`

### Package Installation Issues

If you encounter issues during package installation:

```bash
# Activate the environment
source activate.sh  # or activate.bat on Windows

# Install a specific package manually
pip install package-name
```

### Data Access Issues

If you have trouble accessing the data:

1. Ensure your Google account has access to the shared folder
2. Delete `credentials.json` and run `python simple_setup.py` again
3. Check that the `DATA_PATH` environment variable is set correctly

### GPU Issues

If PyTorch isn't using your GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
```

If CUDA is not available but you have an NVIDIA GPU, reinstall PyTorch:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
