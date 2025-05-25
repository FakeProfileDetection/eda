import polars as pl
import numpy as np
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import json
from typing import Dict, List, Tuple
import pickle

class TypeNetSequentialFeatureExtractor:
    """
    Extract sequential features for TypeNet deep learning experiments
    Features: Key1, Key2, HL, IL, PL, RL (normalized)
    """
    
    def __init__(self, data_path: str = 'typenet_features_extracted.csv'):
        """Initialize with extracted TypeNet features"""
        print(f"Loading data from {data_path}...")
        self.df = pl.read_csv(data_path)
        
        # Filter only valid data
        self.df = self.df.filter(pl.col('valid'))
        
        # Convert timing features to milliseconds
        for col in ['HL', 'IL', 'PL', 'RL']:
            self.df = self.df.with_columns(
                (pl.col(col) / 1_000_000).alias(f'{col}_ms')
            )
        
        print(f"Loaded {len(self.df):,} valid keystroke pairs")
        print(f"Users: {self.df['user_id'].n_unique()}")
        print(f"Platforms: {sorted(self.df['platform_id'].unique().to_list())}")
        
    def create_key_mapping(self) -> Dict[str, int]:
        """
        Create alphabetical mapping of keys to integers
        Returns mapping dictionary
        """
        # Get all unique keys
        all_keys = pl.concat([
            self.df.select('key1'),
            self.df.select(pl.col('key2').alias('key1'))
        ])['key1'].unique().sort().to_list()
        
        # Create mapping (0 to n-1)
        key_to_int = {key: idx for idx, key in enumerate(all_keys)}
        
        print(f"Created key mapping for {len(key_to_int)} unique keys")
        return key_to_int
    
    def normalize_keys(self, key_mapping: Dict[str, int]) -> pl.DataFrame:
        """
        Encode and normalize keys to [0, 1] range
        """
        n_keys = len(key_mapping)
        
        # Create encoded columns using replace_strict
        df = self.df.with_columns([
            pl.col('key1').replace_strict(key_mapping).alias('key1_encoded'),
            pl.col('key2').replace_strict(key_mapping).alias('key2_encoded')
        ])
        
        # Normalize to [0, 1]
        df = df.with_columns([
            (pl.col('key1_encoded') / (n_keys - 1)).alias('key1_normalized'),
            (pl.col('key2_encoded') / (n_keys - 1)).alias('key2_normalized')
        ])
        
        return df
    
    def fit_scalers(self, train_data: pl.DataFrame) -> Dict[str, RobustScaler]:
        """
        Fit RobustScaler for each timing feature on training data
        Returns dictionary of fitted scalers
        """
        scalers = {}
        
        for feature in ['HL_ms', 'IL_ms', 'PL_ms', 'RL_ms']:
            scaler = RobustScaler()
            # Convert to numpy for sklearn
            values = train_data[feature].to_numpy().reshape(-1, 1)
            scaler.fit(values)
            scalers[feature] = scaler
            
            print(f"Fitted scaler for {feature}: center={scaler.center_[0]:.2f}, scale={scaler.scale_[0]:.2f}")
        
        return scalers
    
    def apply_scalers(self, df: pl.DataFrame, scalers: Dict[str, RobustScaler]) -> pl.DataFrame:
        """
        Apply fitted scalers to timing features
        """
        df_scaled = df.clone()
        
        for feature, scaler in scalers.items():
            # Transform using the fitted scaler
            values = df[feature].to_numpy().reshape(-1, 1)
            scaled_values = scaler.transform(values).flatten()
            
            # Add normalized column
            df_scaled = df_scaled.with_columns(
                pl.Series(f'{feature}_normalized', scaled_values)
            )
        
        return df_scaled
    
    def generate_sequential_dataset(self, output_dir: str = 'sequential_datasets'):
        """
        Generate complete sequential dataset with normalization
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating TypeNet Sequential Dataset")
        print("="*60)
        
        # Create key mapping
        key_mapping = self.create_key_mapping()
        
        # Save key mapping
        mapping_info = {
            'key_to_int': key_mapping,
            'int_to_key': {v: k for k, v in key_mapping.items()},
            'n_keys': len(key_mapping)
        }
        
        with open(output_path / 'key_mapping.json', 'w') as f:
            json.dump(mapping_info, f, indent=2)
        
        print(f"\nSaved key mapping to {output_path / 'key_mapping.json'}")
        
        # Encode and normalize keys
        df_encoded = self.normalize_keys(key_mapping)
        
        # Define train data for fitting scalers (using session 1 as default)
        train_data = df_encoded.filter(pl.col('session_id') == 1)
        
        print(f"\nFitting scalers on training data (session 1): {len(train_data)} samples")
        
        # Fit scalers on training data
        scalers = self.fit_scalers(train_data)
        
        # Save scalers for later use
        with open(output_path / 'scalers.pkl', 'wb') as f:
            pickle.dump(scalers, f)
        
        print(f"Saved scalers to {output_path / 'scalers.pkl'}")
        
        # Apply scalers to entire dataset
        df_normalized = self.apply_scalers(df_encoded, scalers)
        
        # Select final columns for sequential dataset
        sequential_columns = [
            'key1_normalized',
            'key2_normalized', 
            'HL_ms_normalized',
            'IL_ms_normalized',
            'PL_ms_normalized',
            'RL_ms_normalized',
            'user_id',
            'platform_id',
            'video_id',
            'session_id',
            'sequence_id'
        ]
        
        # Also include original encoded values for reference
        df_final = df_normalized.select(
            sequential_columns + ['key1_encoded', 'key2_encoded']
        )
        
        # Rename normalized columns to match expected format
        df_final = df_final.rename({
            'key1_normalized': 'Key1',
            'key2_normalized': 'Key2',
            'HL_ms_normalized': 'HL',
            'IL_ms_normalized': 'IL',
            'PL_ms_normalized': 'PL',
            'RL_ms_normalized': 'RL'
        })
        
        # Save complete dataset
        output_file = output_path / 'sequential_typenet_features.csv'
        df_final.write_csv(str(output_file))
        
        print(f"\nSaved sequential dataset to {output_file}")
        print(f"Dataset shape: {df_final.shape}")
        
        # Generate statistics report
        self.generate_sequential_report(df_final, scalers, key_mapping, output_path)
        
        return df_final
    
    def generate_sequential_report(self, df: pl.DataFrame, scalers: Dict, 
                                 key_mapping: Dict, output_path: Path):
        """Generate summary report for sequential dataset"""
        
        report = f"""# TypeNet Sequential Dataset Report

## Dataset Overview
- Total sequences: {len(df):,}
- Unique users: {df['user_id'].n_unique()}
- Platforms: {sorted(df['platform_id'].unique().to_list())}
- Sessions: {sorted(df['session_id'].unique().to_list())}

## Key Encoding
- Total unique keys: {len(key_mapping)}
- Encoding: Alphabetical order → Integer (0 to {len(key_mapping)-1})
- Normalization: Divided by {len(key_mapping)-1} to get [0, 1] range

## Timing Feature Normalization
Using RobustScaler (removes median and scales by IQR)

| Feature | Center (median) | Scale (IQR) |
|---------|----------------|-------------|
"""
        
        for feature, scaler in scalers.items():
            report += f"| {feature} | {scaler.center_[0]:.2f} | {scaler.scale_[0]:.2f} |\n"
        
        report += f"""

## Feature Ranges After Normalization
"""
        
        for col in ['HL', 'IL', 'PL', 'RL']:
            # Get statistics directly
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            col_std = float(df[col].std())
            report += f"\n{col}: min={col_min:.3f}, max={col_max:.3f}, mean={col_mean:.3f}, std={col_std:.3f}"
        
        report += f"""

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
"""
        
        with open(output_path / 'sequential_dataset_report.md', 'w') as f:
            f.write(report)
        
        print(f"Generated report: {output_path / 'sequential_dataset_report.md'}")


# Main execution
if __name__ == "__main__":
    # Initialize extractor
    typenet_features_path = 'processed_data-2025-05-24_144726-Loris-MBP.cable.rcn.com/typenet_features.csv'
    extractor = TypeNetSequentialFeatureExtractor(data_path=typenet_features_path)    
    
    # Generate sequential dataset
    sequential_df = extractor.generate_sequential_dataset('sequential_datasets')
    
    print("\n✅ Sequential dataset generation complete!")