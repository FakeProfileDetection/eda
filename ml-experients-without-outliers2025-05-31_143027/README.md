# TypeNet ML Experiments Summary

## Directory Structure
```
ml-experients-without-outliers2025-05-31_143027/
├── feature_info.json          # List of unigrams, digrams, and feature order
├── imputation_global/         # Global mean imputation
│   ├── dataset_1_full.csv     # User-Platform level features
│   ├── dataset_2_full.csv     # User-Platform-Session level features
│   ├── dataset_3_full.csv     # User-Platform-Session-Video level features
└── imputation_user/           # User-specific mean imputation
    └── ... (same structure)
```

## Dataset Descriptions

### Dataset 1: User-Platform Level
- One feature vector per user per platform
- Aggregates all sessions and videos
- Suitable for experiments where you want platform-level representation

### Dataset 2: User-Platform-Session Level  
- Two feature vectors per user per platform (one per session)
- Aggregates all videos within each session
- Suitable for session-based experiments

### Dataset 3: User-Platform-Session-Video Level
- Six feature vectors per user per platform (3 videos × 2 sessions)
- Most granular level - no aggregation
- Suitable for fine-grained analysis

## Feature Structure
Each feature vector contains:
- Unigram features (HL - Hold Latency): 5 statistics × N unigrams
- Digram features (IL - Inter-key Latency): 5 statistics × 10 top digrams
- Statistics order: median, mean, std, q1, q3

## Experiments

### Session Experiments (1)
- **session_1vs2**: Session 1 (train) vs Session 2 (test)

### Platform 3-choose-2 Experiments (3)
- **platform_3c2_train23_test1**: Train on platforms [2, 3], test on platform 1
- **platform_3c2_train13_test2**: Train on platforms [1, 3], test on platform 2
- **platform_3c2_train12_test3**: Train on platforms [1, 2], test on platform 3

### Platform 3-choose-1 Experiments (6)
- **platform_3c1_train1_test2**: Train on platform 1, test on platform 2
- **platform_3c1_train2_test1**: Train on platform 2, test on platform 1
- **platform_3c1_train1_test3**: Train on platform 1, test on platform 3
- **platform_3c1_train3_test1**: Train on platform 3, test on platform 1
- **platform_3c1_train2_test3**: Train on platform 2, test on platform 3
- **platform_3c1_train3_test2**: Train on platform 3, test on platform 2


## Platform Mapping
- Platform 1: Facebook
- Platform 2: Instagram  
- Platform 3: Twitter

## Usage Example
```python
import polars as pl

# Load a specific experiment
train_data = pl.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_1_train.csv')
test_data = pl.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_1_test.csv')

# Features start from column index 2 (after user_id and platform_id)
X_train = train_data.select(pl.all().exclude(['user_id', 'platform_id'])).to_numpy()
y_train = train_data['user_id'].to_numpy()

X_test = test_data.select(pl.all().exclude(['user_id', 'platform_id'])).to_numpy()
y_test = test_data['user_id'].to_numpy()
```

## Performance Note
This version uses Polars instead of Pandas for significantly faster processing, 
especially beneficial for large datasets with many users and features.
