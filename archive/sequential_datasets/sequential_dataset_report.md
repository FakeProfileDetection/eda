# TypeNet Sequential Dataset Report

## Dataset Overview
- Total sequences: 259,823
- Unique users: 36
- Platforms: [1, 2, 3]
- Sessions: [1, 2]

## Key Encoding
- Total unique keys: 143
- Encoding: Alphabetical order → Integer (0 to 142)
- Normalization: Divided by 142 to get [0, 1] range

## Timing Feature Normalization
Using RobustScaler (removes median and scales by IQR)

| Feature | Center (median) | Scale (IQR) |
|---------|----------------|-------------|
| HL_ms | 90.68 | 44.36 |
| IL_ms | 97.85 | 178.25 |
| PL_ms | 187.65 | 173.89 |
| RL_ms | 191.47 | 179.98 |


## Feature Ranges After Normalization

HL: min=-2.041, max=667.760, mean=0.231, std=2.483
IL: min=-167.109, max=4764.436, mean=1.289, std=17.655
PL: min=-1.077, max=4883.891, mean=1.386, std=18.093
RL: min=-166.014, max=4718.612, mean=1.327, std=17.554

## Files Generated
1. `sequential_typenet_features.csv` - Main dataset
2. `key_mapping.json` - Key to integer mapping
3. `scalers.pkl` - Fitted RobustScaler objects

## Usage Example
```python
import polars as pl
import json
import pickle

# Load dataset
df = pl.read_csv('sequential_typenet_features.csv')

# Load key mapping
with open('key_mapping.json', 'r') as f:
    mapping = json.load(f)

# Load scalers
with open('scalers.pkl', 'rb') as f:
    scalers = pickle.load(f)

# Filter for specific experiment (e.g., session 1 vs 2)
train_data = df.filter(pl.col('session_id') == 1)
test_data = df.filter(pl.col('session_id') == 2)

# Extract sequences for a user
user_sequences = df.filter(pl.col('user_id') == 1001).sort('sequence_id')
```
