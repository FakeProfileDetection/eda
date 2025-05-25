# TypeNet ML Features Dataset Documentation

## Overview

This repository contains machine learning features extracted from TypeNet keystroke dynamics data for user identification experiments. The features are organized into multiple datasets with different aggregation levels and experimental configurations.

## Directory Structure

```
ml_experiments/
├── README.md                      # This file
├── feature_info.json             # Feature specifications and metadata
│
├── imputation_global/            # Global mean imputation strategy
│   ├── dataset_1_full.csv        # User-Platform level features
│   ├── dataset_2_full.csv        # User-Platform-Session level features
│   ├── dataset_3_full.csv        # User-Platform-Session-Video level features
│   │
│   ├── session_1vs2/             # Session-based experiment
│   │   ├── experiment_info.json
│   │   ├── dataset_2_train.csv   # Session 1 data (training)
│   │   ├── dataset_2_test.csv    # Session 2 data (testing)
│   │   ├── dataset_3_train.csv
│   │   └── dataset_3_test.csv
│   │
│   ├── platform_3c2_train12_test3/   # Platform 3-choose-2 experiments
│   │   ├── experiment_info.json      # (Train on 2 platforms, test on 1)
│   │   ├── dataset_1_train.csv
│   │   ├── dataset_1_test.csv
│   │   ├── dataset_2_train.csv
│   │   ├── dataset_2_test.csv
│   │   ├── dataset_3_train.csv
│   │   └── dataset_3_test.csv
│   │
│   ├── platform_3c2_train13_test2/
│   │   └── ... (same structure)
│   │
│   ├── platform_3c2_train23_test1/
│   │   └── ... (same structure)
│   │
│   ├── platform_3c1_train1_test2/    # Platform 3-choose-1 experiments
│   │   └── ... (same structure)      # (Train on 1 platform, test on another)
│   │
│   ├── platform_3c1_train1_test3/
│   │   └── ... (same structure)
│   │
│   ├── platform_3c1_train2_test1/
│   │   └── ... (same structure)
│   │
│   ├── platform_3c1_train2_test3/
│   │   └── ... (same structure)
│   │
│   ├── platform_3c1_train3_test1/
│   │   └── ... (same structure)
│   │
│   └── platform_3c1_train3_test2/
│       └── ... (same structure)
│
└── imputation_user/              # User-specific mean imputation strategy
    └── ... (identical structure to imputation_global/)
```

## Dataset Descriptions

### Dataset Levels

1. **Dataset 1: User-Platform Level**
   - **Aggregation**: All typing data for each user on each platform
   - **Records**: One feature vector per user per platform
   - **Use Case**: Platform-independent user identification
   - **Example**: User 1001 on Facebook → single feature vector combining all their Facebook typing

2. **Dataset 2: User-Platform-Session Level**
   - **Aggregation**: Typing data grouped by session (1 or 2)
   - **Records**: Up to 2 feature vectors per user per platform
   - **Use Case**: Session-aware identification, temporal analysis
   - **Example**: User 1001 on Facebook Session 1 → one feature vector

3. **Dataset 3: User-Platform-Session-Video Level**
   - **Aggregation**: No aggregation - most granular level
   - **Records**: Up to 6 feature vectors per user per platform (2 sessions × 3 videos)
   - **Use Case**: Fine-grained analysis, video-specific patterns
   - **Example**: User 1001 on Facebook Session 1 Video 2 → one feature vector

### Feature Structure

Each feature vector contains:

#### Unigram Features (Hold Latency - HL)
- Features for each unique character in the dataset
- 5 statistics per character: `median`, `mean`, `std`, `q1`, `q3`
- Example: `HL_a_median`, `HL_a_mean`, `HL_a_std`, `HL_a_q1`, `HL_a_q3`

#### Digram Features (Inter-key Latency - IL)
- Features for top 10 most frequent character pairs
- 5 statistics per digram: `median`, `mean`, `std`, `q1`, `q3`
- Example: `IL_th_median`, `IL_he_mean`, etc.

Total features per vector: `(num_unigrams + 10_digrams) × 5_statistics`

### Imputation Strategies

1. **Global Imputation** (`imputation_global/`)
   - Missing values filled with global mean across all users
   - Suitable when assuming population-wide patterns

2. **User Imputation** (`imputation_user/`)
   - Missing values filled with user-specific means
   - Falls back to global mean if user has no data for that feature
   - Better preserves individual typing characteristics

## Experiment Types

### 1. Session Experiments
- **Name**: `session_1vs2`
- **Training**: Session 1 data
- **Testing**: Session 2 data
- **Purpose**: Evaluate temporal stability of typing patterns
- **Note**: Dataset 1 not available (no session information at that level)

### 2. Platform 3-Choose-2 Experiments
- **Names**: `platform_3c2_train[XY]_test[Z]`
- **Training**: Data from 2 platforms
- **Testing**: Data from the remaining platform
- **Configurations**:
  - Train on Facebook & Instagram, test on Twitter
  - Train on Facebook & Twitter, test on Instagram
  - Train on Instagram & Twitter, test on Facebook
- **Purpose**: Cross-platform generalization with substantial training data

### 3. Platform 3-Choose-1 Experiments
- **Names**: `platform_3c1_train[X]_test[Y]`
- **Training**: Data from 1 platform
- **Testing**: Data from another platform
- **Configurations**: All 6 permutations of platform pairs
- **Purpose**: Extreme cross-platform challenge with limited training data

## Platform Mapping
- Platform 1: Facebook
- Platform 2: Instagram
- Platform 3: Twitter

## Usage Examples

### Loading Data with Pandas
```python
import pandas as pd
import numpy as np

# Load a specific experiment
train_df = pd.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_2_train.csv')
test_df = pd.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_2_test.csv')

# Extract features and labels
# Identify ID columns based on dataset type
id_cols = ['user_id', 'platform_id', 'session_id']  # for dataset_2
feature_cols = [col for col in train_df.columns if col not in id_cols]

X_train = train_df[feature_cols].values
y_train = train_df['user_id'].values

X_test = test_df[feature_cols].values
y_test = test_df['user_id'].values
```

### Loading Data with Polars (Faster)
```python
import polars as pl

# Load experiment data
train_df = pl.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_2_train.csv')
test_df = pl.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_2_test.csv')

# Extract features (excluding ID columns)
X_train = train_df.select(pl.all().exclude(['user_id', 'platform_id', 'session_id'])).to_numpy()
y_train = train_df['user_id'].to_numpy()
```

### Loading Feature Information
```python
import json

# Load feature metadata
with open('ml_experiments/feature_info.json', 'r') as f:
    feature_info = json.load(f)

print(f"Number of unigrams: {len(feature_info['unigrams'])}")
print(f"Top digrams: {feature_info['digrams']}")
print(f"Statistics order: {feature_info['feature_order']}")
```

### Implementing Similarity Search
```python
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Compute similarity matrix
similarity_matrix = cosine_similarity(X_test_scaled, X_train_scaled)

# For each test sample, find most similar training samples
for i in range(len(X_test)):
    top_5_indices = np.argsort(similarity_matrix[i])[::-1][:5]
    top_5_users = y_train[top_5_indices]
    print(f"Test user {y_test[i]} → Top 5 predictions: {top_5_users}")
```

## Data Generation Process

### 1. Raw Data Processing
- Started with TypeNet keystroke data: press/release timestamps
- Filtered for valid keystroke pairs only
- Calculated timing features (HL, IL, PL, RL)

### 2. Feature Extraction
- **Unigrams**: All unique characters in dataset
- **Digrams**: Top 10 most frequent character pairs
- **Statistics**: For each unigram/digram, calculated median, mean, std, q1, q3

### 3. Aggregation Levels
- **Dataset 1**: Aggregated across all sessions and videos per user-platform
- **Dataset 2**: Aggregated across videos within each session
- **Dataset 3**: No aggregation - raw granularity

### 4. Missing Data Handling
- **Global**: Replace NaN with mean across all users
- **User**: Replace NaN with user's mean, fallback to global mean

### 5. Train-Test Splitting
- Based on experiment configuration
- Maintains user consistency (same user doesn't appear in both train and test)

## Performance Considerations

- **Polars version**: ~47% faster than Pandas version
- **File sizes**: Dataset 3 > Dataset 2 > Dataset 1
- **Memory usage**: Load datasets incrementally for large-scale experiments

## Evaluation Metrics

The datasets are designed for similarity-based user identification:
- **Top-1 Accuracy**: Correct user is the most similar
- **Top-3 Accuracy**: Correct user in top 3 most similar
- **Top-5 Accuracy**: Correct user in top 5 most similar

## Citation

If using this dataset, please cite:
```
Acien, A., Morales, A., Monaco, J. V., Vera-Rodriguez, R., & Fierrez, J. (2021). 
TypeNet: Deep Learning Keystroke Biometrics. 
IEEE Transactions on Biometrics, Behavior, and Identity Science.
```

## Contact

For questions about the dataset structure or usage, please refer to the generation scripts:
- `typenet_ml_features.py` (Pandas version)
- `typenet_ml_features_polars.py` (Polars version - recommended)


