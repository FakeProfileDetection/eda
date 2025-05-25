import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

class TypeNetMLFeatureExtractor:
    """
    Feature extraction system for TypeNet ML experiments
    Handles multiple experiment types and feature extraction strategies
    """
    
    def __init__(self, data_path: str = 'typenet_features_extracted.csv'):
        """Initialize with extracted TypeNet features"""
        print(f"Loading data from {data_path}...")
        self.df = pd.read_csv(data_path)
        
        # Filter only valid data for ML
        self.df = self.df[self.df['valid']].copy()
        
        # Convert to milliseconds
        for col in ['HL', 'IL', 'PL', 'RL']:
            self.df[f'{col}_ms'] = self.df[col] / 1_000_000
        
        self.platform_names = {1: 'facebook', 2: 'instagram', 3: 'twitter'}
        
        print(f"Loaded {len(self.df):,} valid keystroke pairs")
        print(f"Users: {self.df['user_id'].nunique()}")
        print(f"Platforms: {sorted(self.df['platform_id'].unique())}")
        print(f"Sessions per platform: {self.df.groupby('platform_id')['session_id'].nunique().to_dict()}")
        
    def get_top_digrams(self, n: int = 10) -> List[str]:
        """Get top N most frequent digrams across entire dataset"""
        self.df['digram'] = self.df['key1'] + self.df['key2']
        digram_counts = self.df['digram'].value_counts()
        top_digrams = digram_counts.head(n).index.tolist()
        print(f"\nTop {n} digrams: {top_digrams}")
        return top_digrams
    
    def get_all_unigrams(self) -> List[str]:
        """Get all unique unigrams (individual keys) in dataset"""
        all_keys = pd.concat([self.df['key1'], self.df['key2']]).unique()
        unigrams = sorted(list(all_keys))
        print(f"\nTotal unique unigrams: {len(unigrams)}")
        return unigrams
    
    def extract_features(self, data: pd.DataFrame, unigrams: List[str], 
                        digrams: List[str]) -> Dict[str, float]:
        """
        Extract statistical features for given data subset
        Returns: Dict with features in order: median, mean, std, q1, q3
        """
        features = {}
        
        # Extract unigram (HL) features
        for unigram in unigrams:
            # Filter data for this unigram
            unigram_data = data[data['key1'] == unigram]['HL_ms']
            
            if len(unigram_data) > 0:
                features[f'HL_{unigram}_median'] = unigram_data.median()
                features[f'HL_{unigram}_mean'] = unigram_data.mean()
                features[f'HL_{unigram}_std'] = unigram_data.std() if len(unigram_data) > 1 else 0
                features[f'HL_{unigram}_q1'] = unigram_data.quantile(0.25)
                features[f'HL_{unigram}_q3'] = unigram_data.quantile(0.75)
            else:
                # Missing data - will be handled by imputation strategy
                for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                    features[f'HL_{unigram}_{stat}'] = np.nan
        
        # Extract digram (IL) features
        for digram in digrams:
            # Filter data for this digram
            digram_data = data[data['digram'] == digram]['IL_ms']
            
            if len(digram_data) > 0:
                features[f'IL_{digram}_median'] = digram_data.median()
                features[f'IL_{digram}_mean'] = digram_data.mean()
                features[f'IL_{digram}_std'] = digram_data.std() if len(digram_data) > 1 else 0
                features[f'IL_{digram}_q1'] = digram_data.quantile(0.25)
                features[f'IL_{digram}_q3'] = digram_data.quantile(0.75)
            else:
                # Missing data - will be handled by imputation strategy
                for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                    features[f'IL_{digram}_{stat}'] = np.nan
        
        return features
    
    def create_dataset_1(self, unigrams: List[str], digrams: List[str], 
                        imputation: str = 'global') -> pd.DataFrame:
        """
        Dataset 1: One set of features per user/platform
        Aggregates all sessions and videos for each user-platform combination
        """
        print("\nCreating Dataset 1: User-Platform level features...")
        
        feature_records = []
        
        # Group by user and platform
        for (user_id, platform_id), group_data in self.df.groupby(['user_id', 'platform_id']):
            features = self.extract_features(group_data, unigrams, digrams)
            features['user_id'] = user_id
            features['platform_id'] = platform_id
            feature_records.append(features)
        
        # Create DataFrame
        dataset = pd.DataFrame(feature_records)
        
        # Apply imputation strategy
        dataset = self.apply_imputation(dataset, imputation, level='platform')
        
        # Reorder columns
        id_cols = ['user_id', 'platform_id']
        feature_cols = [col for col in dataset.columns if col not in id_cols]
        dataset = dataset[id_cols + sorted(feature_cols)]
        
        print(f"Dataset 1 shape: {dataset.shape}")
        return dataset
    
    def create_dataset_2(self, unigrams: List[str], digrams: List[str], 
                        imputation: str = 'global') -> pd.DataFrame:
        """
        Dataset 2: Two sets of features per user/platform/session
        Aggregates all videos for each user-platform-session combination
        """
        print("\nCreating Dataset 2: User-Platform-Session level features...")
        
        feature_records = []
        
        # Group by user, platform, and session
        for (user_id, platform_id, session_id), group_data in self.df.groupby(['user_id', 'platform_id', 'session_id']):
            features = self.extract_features(group_data, unigrams, digrams)
            features['user_id'] = user_id
            features['platform_id'] = platform_id
            features['session_id'] = session_id
            feature_records.append(features)
        
        # Create DataFrame
        dataset = pd.DataFrame(feature_records)
        
        # Apply imputation strategy
        dataset = self.apply_imputation(dataset, imputation, level='session')
        
        # Reorder columns
        id_cols = ['user_id', 'platform_id', 'session_id']
        feature_cols = [col for col in dataset.columns if col not in id_cols]
        dataset = dataset[id_cols + sorted(feature_cols)]
        
        print(f"Dataset 2 shape: {dataset.shape}")
        return dataset
    
    def create_dataset_3(self, unigrams: List[str], digrams: List[str], 
                        imputation: str = 'global') -> pd.DataFrame:
        """
        Dataset 3: Six sets of features per user/platform/session/video
        Most granular level - no aggregation
        """
        print("\nCreating Dataset 3: User-Platform-Session-Video level features...")
        
        feature_records = []
        
        # Group by user, platform, session, and video
        for (user_id, platform_id, session_id, video_id), group_data in self.df.groupby(['user_id', 'platform_id', 'session_id', 'video_id']):
            features = self.extract_features(group_data, unigrams, digrams)
            features['user_id'] = user_id
            features['platform_id'] = platform_id
            features['session_id'] = session_id
            features['video_id'] = video_id
            feature_records.append(features)
        
        # Create DataFrame
        dataset = pd.DataFrame(feature_records)
        
        # Apply imputation strategy
        dataset = self.apply_imputation(dataset, imputation, level='video')
        
        # Reorder columns
        id_cols = ['user_id', 'platform_id', 'session_id', 'video_id']
        feature_cols = [col for col in dataset.columns if col not in id_cols]
        dataset = dataset[id_cols + sorted(feature_cols)]
        
        print(f"Dataset 3 shape: {dataset.shape}")
        return dataset
    
    def apply_imputation(self, dataset: pd.DataFrame, strategy: str, level: str) -> pd.DataFrame:
        """
        Apply imputation strategy for missing values
        strategy: 'global' (average over all users) or 'user' (average over user's data)
        level: 'platform', 'session', or 'video'
        """
        feature_cols = [col for col in dataset.columns if col not in ['user_id', 'platform_id', 'session_id', 'video_id']]
        
        if strategy == 'global':
            # Replace NaN with global mean
            for col in feature_cols:
                global_mean = dataset[col].mean()
                dataset[col].fillna(global_mean, inplace=True)
                
        elif strategy == 'user':
            # Replace NaN with user-specific mean
            for col in feature_cols:
                # First try user-level mean
                user_means = dataset.groupby('user_id')[col].transform('mean')
                dataset[col].fillna(user_means, inplace=True)
                
                # If still NaN (user has no data for this feature), use global mean
                global_mean = dataset[col].mean()
                dataset[col].fillna(global_mean, inplace=True)
        
        # Final check - if still any NaN (e.g., all values were NaN), fill with 0
        dataset[feature_cols] = dataset[feature_cols].fillna(0)
        
        return dataset
    
    def create_experiment_splits(self, dataset: pd.DataFrame, experiment_type: str, 
                               experiment_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create train/test splits based on experiment configuration
        """
        if experiment_type == 'session':
            # Session 1 vs Session 2
            train_data = dataset[dataset['session_id'] == 1].copy()
            test_data = dataset[dataset['session_id'] == 2].copy()
            
        elif experiment_type == 'platform_3c2':
            # 3-choose-2 platform experiments
            train_platforms = experiment_config['train_platforms']
            test_platform = experiment_config['test_platform']
            
            train_data = dataset[dataset['platform_id'].isin(train_platforms)].copy()
            test_data = dataset[dataset['platform_id'] == test_platform].copy()
            
        elif experiment_type == 'platform_3c1':
            # 3-choose-1 platform experiments
            train_platform = experiment_config['train_platform']
            test_platform = experiment_config['test_platform']
            
            train_data = dataset[dataset['platform_id'] == train_platform].copy()
            test_data = dataset[dataset['platform_id'] == test_platform].copy()
        
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        return train_data, test_data
    
    def generate_all_experiments(self, output_dir: str = 'ml_experiments'):
        """
        Generate all datasets for all experiment configurations
        """
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get features
        unigrams = self.get_all_unigrams()
        digrams = self.get_top_digrams(n=10)
        
        # Save feature lists for reference
        feature_info = {
            'unigrams': unigrams,
            'digrams': digrams,
            'feature_order': ['median', 'mean', 'std', 'q1', 'q3']
        }
        with open(output_path / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Define all experiments
        experiments = []
        
        # Session experiments
        experiments.append({
            'name': 'session_1vs2',
            'type': 'session',
            'description': 'Session 1 (train) vs Session 2 (test)',
            'config': {}
        })
        
        # 3-choose-2 platform experiments
        platforms = [1, 2, 3]
        for test_platform in platforms:
            train_platforms = [p for p in platforms if p != test_platform]
            experiments.append({
                'name': f'platform_3c2_train{train_platforms[0]}{train_platforms[1]}_test{test_platform}',
                'type': 'platform_3c2',
                'description': f'Train on platforms {train_platforms}, test on platform {test_platform}',
                'config': {
                    'train_platforms': train_platforms,
                    'test_platform': test_platform
                }
            })
        
        # 3-choose-1 platform experiments
        for train_platform, test_platform in combinations(platforms, 2):
            experiments.append({
                'name': f'platform_3c1_train{train_platform}_test{test_platform}',
                'type': 'platform_3c1',
                'description': f'Train on platform {train_platform}, test on platform {test_platform}',
                'config': {
                    'train_platform': train_platform,
                    'test_platform': test_platform
                }
            })
            
            # Also the reverse
            experiments.append({
                'name': f'platform_3c1_train{test_platform}_test{train_platform}',
                'type': 'platform_3c1',
                'description': f'Train on platform {test_platform}, test on platform {train_platform}',
                'config': {
                    'train_platform': test_platform,
                    'test_platform': train_platform
                }
            })
        
        # Generate datasets with both imputation strategies
        for imputation in ['global', 'user']:
            imp_dir = output_path / f'imputation_{imputation}'
            imp_dir.mkdir(exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Generating datasets with {imputation} imputation")
            print(f"{'='*60}")
            
            # Create all three dataset types
            datasets = {
                'dataset_1': self.create_dataset_1(unigrams, digrams, imputation),
                'dataset_2': self.create_dataset_2(unigrams, digrams, imputation),
                'dataset_3': self.create_dataset_3(unigrams, digrams, imputation)
            }
            
            # Save full datasets
            for dataset_name, dataset in datasets.items():
                dataset.to_csv(imp_dir / f'{dataset_name}_full.csv', index=False)
            
            # Generate experiment splits
            for experiment in experiments:
                print(f"\nProcessing experiment: {experiment['name']}")
                exp_dir = imp_dir / experiment['name']
                exp_dir.mkdir(exist_ok=True)
                
                # Save experiment info
                with open(exp_dir / 'experiment_info.json', 'w') as f:
                    json.dump(experiment, f, indent=2)
                
                # Generate splits for each dataset type
                for dataset_name, dataset in datasets.items():
                    # Skip dataset 3 for session experiments if only one session
                    if experiment['type'] == 'session' and dataset_name == 'dataset_3':
                        if 'session_id' in dataset.columns:
                            unique_sessions = dataset['session_id'].nunique()
                            if unique_sessions < 2:
                                print(f"  Skipping {dataset_name} - insufficient sessions")
                                continue
                    
                    try:
                        train_data, test_data = self.create_experiment_splits(
                            dataset, experiment['type'], experiment['config']
                        )
                        
                        # Save train/test splits
                        train_data.to_csv(exp_dir / f'{dataset_name}_train.csv', index=False)
                        test_data.to_csv(exp_dir / f'{dataset_name}_test.csv', index=False)
                        
                        print(f"  {dataset_name}: Train shape {train_data.shape}, Test shape {test_data.shape}")
                        
                    except Exception as e:
                        print(f"  Error processing {dataset_name}: {e}")
        
        # Generate summary report
        self.generate_summary_report(output_path, experiments)
        
        print(f"\n✅ All experiments generated in '{output_dir}' directory")
        print(f"Total experiments: {len(experiments)}")
        print(f"Dataset types: 3 (user-platform, user-platform-session, user-platform-session-video)")
        print(f"Imputation strategies: 2 (global, user)")
    
    def generate_summary_report(self, output_path: Path, experiments: List[Dict]):
        """Generate a summary report of all experiments"""
        
        report = f"""# TypeNet ML Experiments Summary

## Directory Structure
```
{output_path.name}/
├── feature_info.json          # List of unigrams, digrams, and feature order
├── imputation_global/         # Global mean imputation
│   ├── dataset_1_full.csv     # User-Platform level features
│   ├── dataset_2_full.csv     # User-Platform-Session level features
│   ├── dataset_3_full.csv     # User-Platform-Session-Video level features
│   └── [experiment_name]/     # Each experiment directory
│       ├── experiment_info.json
│       ├── dataset_1_train.csv
│       ├── dataset_1_test.csv
│       ├── dataset_2_train.csv
│       ├── dataset_2_test.csv
│       ├── dataset_3_train.csv
│       └── dataset_3_test.csv
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

"""
        
        # Group experiments by type
        session_exps = [e for e in experiments if e['type'] == 'session']
        platform_3c2_exps = [e for e in experiments if e['type'] == 'platform_3c2']
        platform_3c1_exps = [e for e in experiments if e['type'] == 'platform_3c1']
        
        report += f"### Session Experiments ({len(session_exps)})\n"
        for exp in session_exps:
            report += f"- **{exp['name']}**: {exp['description']}\n"
        
        report += f"\n### Platform 3-choose-2 Experiments ({len(platform_3c2_exps)})\n"
        for exp in platform_3c2_exps:
            report += f"- **{exp['name']}**: {exp['description']}\n"
        
        report += f"\n### Platform 3-choose-1 Experiments ({len(platform_3c1_exps)})\n"
        for exp in platform_3c1_exps:
            report += f"- **{exp['name']}**: {exp['description']}\n"
        
        report += """

## Platform Mapping
- Platform 1: Facebook
- Platform 2: Instagram  
- Platform 3: Twitter

## Usage Example
```python
import pandas as pd

# Load a specific experiment
train_data = pd.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_1_train.csv')
test_data = pd.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_1_test.csv')

# Features start from column index 2 (after user_id and platform_id)
X_train = train_data.iloc[:, 2:].values
y_train = train_data['user_id'].values

X_test = test_data.iloc[:, 2:].values
y_test = test_data['user_id'].values
```
"""
        
        with open(output_path / 'README.md', 'w') as f:
            f.write(report)


# Main execution
if __name__ == "__main__":
    # Initialize feature extractor
    extractor = TypeNetMLFeatureExtractor('typenet_features_extracted.csv')
    
    # Generate all experiments
    extractor.generate_all_experiments('ml_experiments')
    
    print("\n📊 Next steps:")
    print("1. Review ml_experiments/README.md for experiment details")
    print("2. Check ml_experiments/feature_info.json for feature specifications")
    print("3. Load train/test datasets for your ML experiments")
    print("4. Implement similarity search and top-k accuracy evaluation")