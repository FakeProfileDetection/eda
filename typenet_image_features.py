import polars as pl
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import h5py
import warnings
warnings.filterwarnings('ignore')

class TypeNetImageFeatureExtractor:
    """
    Extract image features for TypeNet CNN experiments
    Creates multi-channel 2D arrays (tensors) for each user's typing patterns
    """
    
    def __init__(self, data_path: str = 'typenet_features_extracted.csv'):
        """Initialize with extracted TypeNet features"""
        print(f"Loading data from {data_path}...")
        self.df = pl.read_csv(data_path)
        
        # Filter only valid data
        self.df = self.df.filter(pl.col('valid'))
        
        # Convert to milliseconds
        for col in ['HL', 'IL']:
            self.df = self.df.with_columns(
                (pl.col(col) / 1_000_000).alias(f'{col}_ms')
            )
        
        # Add digram column
        self.df = self.df.with_columns(
            (pl.col('key1') + pl.col('key2')).alias('digram')
        )
        
        print(f"Loaded {len(self.df):,} valid keystroke pairs")
        print(f"Users: {self.df['user_id'].n_unique()}")
        
    def get_all_keys(self) -> List[str]:
        """Get all unique keys sorted alphabetically"""
        all_keys = pl.concat([
            self.df.select('key1'),
            self.df.select(pl.col('key2').alias('key1'))
        ])['key1'].unique().sort().to_list()
        
        return all_keys
    
    def create_feature_grid(self, data: pl.DataFrame, keys: List[str], 
                          feature_type: str, statistic: str) -> np.ndarray:
        """
        Create a 2D grid for a specific feature and statistic
        
        Args:
            data: Subset of data for specific user/platform/session/video
            keys: List of all possible keys
            feature_type: 'HL' or 'IL' 
            statistic: 'median', 'mean', 'std', 'q1', 'q3'
        
        Returns:
            2D numpy array of shape (n_keys, n_keys)
        """
        n_keys = len(keys)
        grid = np.full((n_keys, n_keys), np.nan, dtype=np.float32)
        
        # Create key to index mapping
        key_to_idx = {key: idx for idx, key in enumerate(keys)}
        
        if feature_type == 'HL':
            # HL is based on key1 only (diagonal elements)
            for key in keys:
                if key in key_to_idx:
                    key_data = data.filter(pl.col('key1') == key)['HL_ms']
                    
                    if len(key_data) > 0:
                        idx = key_to_idx[key]
                        
                        if statistic == 'median':
                            grid[idx, idx] = float(key_data.median())
                        elif statistic == 'mean':
                            grid[idx, idx] = float(key_data.mean())
                        elif statistic == 'std':
                            grid[idx, idx] = float(key_data.std()) if len(key_data) > 1 else 0.0
                        elif statistic == 'q1':
                            grid[idx, idx] = float(key_data.quantile(0.25))
                        elif statistic == 'q3':
                            grid[idx, idx] = float(key_data.quantile(0.75))
        
        elif feature_type == 'IL':
            # IL is based on key1-key2 pairs (full matrix)
            for key1 in keys:
                for key2 in keys:
                    if key1 in key_to_idx and key2 in key_to_idx:
                        pair_data = data.filter(
                            (pl.col('key1') == key1) & (pl.col('key2') == key2)
                        )['IL_ms']
                        
                        if len(pair_data) > 0:
                            idx1 = key_to_idx[key1]
                            idx2 = key_to_idx[key2]
                            
                            if statistic == 'median':
                                grid[idx1, idx2] = float(pair_data.median())
                            elif statistic == 'mean':
                                grid[idx1, idx2] = float(pair_data.mean())
                            elif statistic == 'std':
                                grid[idx1, idx2] = float(pair_data.std()) if len(pair_data) > 1 else 0.0
                            elif statistic == 'q1':
                                grid[idx1, idx2] = float(pair_data.quantile(0.25))
                            elif statistic == 'q3':
                                grid[idx1, idx2] = float(pair_data.quantile(0.75))
        
        return grid
    
    def create_multi_channel_image(self, data: pl.DataFrame, keys: List[str], 
                                 fill_value: Optional[float] = None) -> np.ndarray:
        """
        Create multi-channel image for given data
        
        Returns:
            Array of shape (5, n_keys, 2*n_keys) where:
            - 5 channels: median, mean, std, q1, q3
            - Height: n_keys
            - Width: 2*n_keys (HL:IL concatenated)
        """
        n_keys = len(keys)
        statistics = ['median', 'mean', 'std', 'q1', 'q3']
        
        # Initialize multi-channel image
        image = np.zeros((5, n_keys, 2 * n_keys), dtype=np.float32)
        
        for stat_idx, statistic in enumerate(statistics):
            # Create HL grid (n_keys x n_keys)
            hl_grid = self.create_feature_grid(data, keys, 'HL', statistic)
            
            # Create IL grid (n_keys x n_keys)
            il_grid = self.create_feature_grid(data, keys, 'IL', statistic)
            
            # Handle missing values
            if fill_value is not None:
                hl_grid = np.nan_to_num(hl_grid, nan=fill_value)
                il_grid = np.nan_to_num(il_grid, nan=fill_value)
            
            # Concatenate HL and IL horizontally
            concatenated = np.concatenate([hl_grid, il_grid], axis=1)
            
            # Store in appropriate channel
            image[stat_idx] = concatenated
        
        return image
    
    def calculate_user_means(self, user_id: int, keys: List[str]) -> Dict[str, float]:
        """Calculate mean values for each feature/statistic combination for a user"""
        user_data = self.df.filter(pl.col('user_id') == user_id)
        
        means = {}
        statistics = ['median', 'mean', 'std', 'q1', 'q3']
        
        for stat in statistics:
            # Calculate overall means for this statistic
            if stat == 'median':
                means[f'HL_{stat}'] = float(user_data['HL_ms'].median())
                means[f'IL_{stat}'] = float(user_data['IL_ms'].median())
            elif stat == 'mean':
                means[f'HL_{stat}'] = float(user_data['HL_ms'].mean())
                means[f'IL_{stat}'] = float(user_data['IL_ms'].mean())
            elif stat == 'std':
                means[f'HL_{stat}'] = float(user_data['HL_ms'].std())
                means[f'IL_{stat}'] = float(user_data['IL_ms'].std())
            elif stat == 'q1':
                means[f'HL_{stat}'] = float(user_data['HL_ms'].quantile(0.25))
                means[f'IL_{stat}'] = float(user_data['IL_ms'].quantile(0.25))
            elif stat == 'q3':
                means[f'HL_{stat}'] = float(user_data['HL_ms'].quantile(0.75))
                means[f'IL_{stat}'] = float(user_data['IL_ms'].quantile(0.75))
        
        return means
    
    def generate_image_datasets(self, output_dir: str = 'image_datasets'):
        """Generate both mean-fill and zero-fill image datasets"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*60)
        print("Generating TypeNet CNN Image Datasets")
        print("="*60)
        
        # Get all keys
        keys = self.get_all_keys()
        print(f"\nUsing {len(keys)} unique keys")
        print(f"Image dimensions: 5 channels × {len(keys)} × {2*len(keys)}")
        
        # Save key information
        key_info = {
            'keys': keys,
            'n_keys': len(keys),
            'key_to_index': {key: idx for idx, key in enumerate(keys)},
            'channels': ['median', 'mean', 'std', 'q1', 'q3'],
            'image_shape': [5, len(keys), 2 * len(keys)],
            'layout': 'CHW (Channels, Height, Width)',
            'structure': 'HL:IL (concatenated horizontally)'
        }
        
        with open(output_path / 'image_metadata.json', 'w') as f:
            json.dump(key_info, f, indent=2)
        
        # Create directories for both fill strategies
        for fill_strategy in ['mean_fill', 'zero_fill']:
            strategy_dir = output_path / f'image_typenet_features_{fill_strategy}'
            strategy_dir.mkdir(exist_ok=True)
            
            print(f"\nGenerating {fill_strategy} dataset...")
            
            # Get unique combinations
            unique_combos = (
                self.df.select(['user_id', 'platform_id', 'session_id', 'video_id'])
                .unique()
                .sort(['user_id', 'platform_id', 'session_id', 'video_id'])
            )
            
            total_images = len(unique_combos)
            print(f"Total images to generate: {total_images}")
            
            # Process each user
            user_ids = unique_combos['user_id'].unique().sort().to_list()
            
            for user_idx, user_id in enumerate(user_ids):
                # Create user directory
                user_dir = strategy_dir / f'user_{user_id}'
                user_dir.mkdir(exist_ok=True)
                
                # Get user's combinations
                user_combos = unique_combos.filter(pl.col('user_id') == user_id)
                
                # Calculate user means if using mean fill
                if fill_strategy == 'mean_fill':
                    user_means = self.calculate_user_means(user_id, keys)
                
                # Process each combination for this user
                for row in user_combos.iter_rows(named=True):
                    platform_id = row['platform_id']
                    session_id = row['session_id']
                    video_id = row['video_id']
                    
                    # Get data for this combination
                    subset_data = self.df.filter(
                        (pl.col('user_id') == user_id) &
                        (pl.col('platform_id') == platform_id) &
                        (pl.col('session_id') == session_id) &
                        (pl.col('video_id') == video_id)
                    )
                    
                    if len(subset_data) == 0:
                        continue
                    
                    # Create image with appropriate fill strategy
                    if fill_strategy == 'mean_fill':
                        # Create initial image with NaN
                        image = self.create_multi_channel_image(subset_data, keys, fill_value=None)
                        
                        # Fill NaN values with user-specific means per channel
                        for channel_idx, statistic in enumerate(['median', 'mean', 'std', 'q1', 'q3']):
                            # HL part (left half)
                            hl_mean = user_means.get(f'HL_{statistic}', 0)
                            mask = np.isnan(image[channel_idx, :, :len(keys)])
                            image[channel_idx, :, :len(keys)][mask] = hl_mean
                            
                            # IL part (right half)
                            il_mean = user_means.get(f'IL_{statistic}', 0)
                            mask = np.isnan(image[channel_idx, :, len(keys):])
                            image[channel_idx, :, len(keys):][mask] = il_mean
                    else:
                        # Zero fill
                        image = self.create_multi_channel_image(subset_data, keys, fill_value=0.0)
                    
                    # Save as HDF5 for efficient storage and loading
                    filename = f'{platform_id}_{video_id}_{session_id}_{user_id}.h5'
                    filepath = user_dir / filename
                    
                    with h5py.File(filepath, 'w') as hf:
                        hf.create_dataset('image', data=image, compression='gzip', compression_opts=9)
                        # Store metadata
                        hf.attrs['user_id'] = user_id
                        hf.attrs['platform_id'] = platform_id
                        hf.attrs['session_id'] = session_id
                        hf.attrs['video_id'] = video_id
                        hf.attrs['shape'] = image.shape
                        hf.attrs['fill_strategy'] = fill_strategy
                
                if (user_idx + 1) % 10 == 0:
                    print(f"  Processed {user_idx + 1}/{len(user_ids)} users")
            
            print(f"✓ Completed {fill_strategy} dataset")
            
            # Generate dataset statistics
            self.generate_dataset_stats(strategy_dir, fill_strategy, output_path)
        
        # Generate usage guide
        self.generate_usage_guide(output_path)
        
        print(f"\n✅ Image datasets generation complete!")
        print(f"Output directory: {output_path}")
    
    def generate_dataset_stats(self, dataset_dir: Path, fill_strategy: str, output_path: Path):
        """Generate statistics for the dataset"""
        total_files = sum(1 for _ in dataset_dir.rglob('*.h5'))
        total_size = sum(f.stat().st_size for f in dataset_dir.rglob('*.h5'))
        
        stats = {
            'fill_strategy': fill_strategy,
            'total_images': total_files,
            'total_size_mb': total_size / (1024 * 1024),
            'avg_size_kb': (total_size / total_files / 1024) if total_files > 0 else 0
        }
        
        with open(output_path / f'{fill_strategy}_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
    
    def generate_usage_guide(self, output_path: Path):
        """Generate usage guide for the image datasets"""
        
        guide = """# TypeNet CNN Image Dataset Usage Guide

## Dataset Structure
```
image_datasets/
├── image_metadata.json           # Key mapping and image specifications
├── mean_fill_stats.json         # Dataset statistics (mean fill)
├── zero_fill_stats.json         # Dataset statistics (zero fill)
├── image_typenet_features_mean_fill/
│   └── user_[id]/
│       └── [platform]_[video]_[session]_[user].h5
└── image_typenet_features_zero_fill/
    └── user_[id]/
        └── [platform]_[video]_[session]_[user].h5
```

## Image Format
- **Storage**: HDF5 with gzip compression
- **Shape**: (5, n_keys, 2*n_keys)
- **Channels**: [median, mean, std, q1, q3]
- **Layout**: HL matrix | IL matrix (concatenated horizontally)
- **Data type**: float32

## Loading Images

### Single Image
```python
import h5py
import torch

# Load single image
with h5py.File('path/to/image.h5', 'r') as f:
    image = f['image'][:]  # Shape: (5, n_keys, 2*n_keys)
    # Metadata
    user_id = f.attrs['user_id']
    platform_id = f.attrs['platform_id']
    
# Convert to PyTorch tensor
tensor = torch.from_numpy(image).float()
```

### Batch Loading
```python
from pathlib import Path
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class TypeNetImageDataset(Dataset):
    def __init__(self, root_dir, fill_strategy='mean_fill'):
        self.root_dir = Path(root_dir) / f'image_typenet_features_{fill_strategy}'
        self.image_files = list(self.root_dir.rglob('*.h5'))
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        with h5py.File(self.image_files[idx], 'r') as f:
            image = torch.from_numpy(f['image'][:]).float()
            user_id = f.attrs['user_id']
        return image, user_id

# Create DataLoader
dataset = TypeNetImageDataset('image_datasets', 'mean_fill')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Experiment Splits
```python
# Session experiment
train_files = [f for f in Path('image_datasets/image_typenet_features_mean_fill').rglob('*.h5') 
               if '_1_' in f.stem]  # Session 1
test_files = [f for f in Path('image_datasets/image_typenet_features_mean_fill').rglob('*.h5') 
              if '_2_' in f.stem]  # Session 2

# Platform experiment (e.g., platforms 1&2 vs 3)
train_files = [f for f in Path('image_datasets/image_typenet_features_mean_fill').rglob('*.h5') 
               if f.stem.startswith(('1_', '2_'))]
test_files = [f for f in Path('image_datasets/image_typenet_features_mean_fill').rglob('*.h5') 
              if f.stem.startswith('3_')]
```

## Fill Strategies
1. **mean_fill**: Missing values filled with user-specific means per feature/statistic
2. **zero_fill**: Missing values filled with 0

## Preparing for GCS Upload
```bash
# Compress for upload
cd image_datasets
tar -czf image_typenet_features_mean_fill.tar.gz image_typenet_features_mean_fill/
tar -czf image_typenet_features_zero_fill.tar.gz image_typenet_features_zero_fill/

# Upload to GCS
gsutil cp image_typenet_features_*.tar.gz gs://your-bucket/typenet/
```

## Notes
- HDF5 with gzip-9 provides ~70% compression
- Images are stored per user directory for efficient access
- Metadata is embedded in each HDF5 file
- Float32 precision balances accuracy and storage
"""
        
        with open(output_path / 'USAGE_GUIDE.md', 'w') as f:
            f.write(guide)


# Main execution
if __name__ == "__main__":
    # Initialize extractor
    typenet_features_path = 'processed_data-2025-05-24_144726-Loris-MBP.cable.rcn.com/typenet_features.csv'
    extractor = TypeNetImageFeatureExtractor(data_path=typenet_features_path)    
    
    # Generate image datasets
    extractor.generate_image_datasets('image_datasets')
    
    print("\n📊 Next steps:")
    print("1. Review image_datasets/USAGE_GUIDE.md for loading instructions")
    print("2. Check image_datasets/*_stats.json for dataset statistics")
    print("3. Compress datasets for GCS upload using tar")
    print("4. Implement PyTorch Dataset class for training")