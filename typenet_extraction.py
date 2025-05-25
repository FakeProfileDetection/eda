import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
import socket
from datetime import datetime

class HBOS:
    def __init__(self, n_bins=10, contamination=0.1, alpha=0.1, tol=0.5):
        """
        HBOS with smoothing (alpha) and edge-tolerance (tol).
        Parameters:
        - n_bins: Base number of bins per feature (can also be 'auto' rules).
        - contamination: Proportion of expected outliers.
        - alpha: Small constant added to every bin density.
        - tol: Fraction of one bin-width to tolerate just-outside values.
        """
        self.n_bins = n_bins
        self.contamination = contamination
        self.alpha = alpha
        self.tol = tol
        self.histograms = []  # List of 1D arrays: per-feature bin densities
        self.bin_edges = []   # List of 1D arrays: per-feature edges
        self.feature_names = []  # Keys order
        
    def fit(self, data: defaultdict):
        """Build (smoothed) histograms for each feature."""
        self.feature_names = list(data.keys())
        X = np.column_stack([data[f] for f in self.feature_names])
        self.histograms.clear()
        self.bin_edges.clear()
        
        for col in X.T:
            # 1) build raw histogram
            hist, edges = np.histogram(col, bins=self.n_bins, density=True)
            # 2) smooth: add alpha everywhere
            hist = hist + self.alpha
            self.histograms.append(hist)
            self.bin_edges.append(edges)
            
    def _compute_score(self, x: np.ndarray) -> float:
        """
        Negative-log-sum of per-feature densities with alpha & tol handling.
        Higher score = more anomalous.
        """
        score = 0.0
        for i, xi in enumerate(x):
            edges = self.bin_edges[i]
            hist = self.histograms[i]
            n_bins = hist.shape[0]
            
            # compute first/last bin widths
            width_low = edges[1] - edges[0]
            width_high = edges[-1] - edges[-2]
            
            # 1) too far below range?
            if xi < edges[0]:
                if edges[0] - xi <= self.tol * width_low:
                    # snap into first bin
                    density = hist[0]
                else:
                    # true out-of-range → worst density
                    density = self.alpha
                score += -np.log(density)
                continue
                
            # 2) too far above range?
            if xi > edges[-1]:
                if xi - edges[-1] <= self.tol * width_high:
                    # snap into last bin
                    density = hist[-1]
                else:
                    density = self.alpha
                score += -np.log(density)
                continue
                
            # 3) within [min, max] → find bin index
            bin_idx = np.searchsorted(edges, xi, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            density = hist[bin_idx]
            score += -np.log(density)
            
        return score
        
    def decision_function(self, data: defaultdict) -> np.ndarray:
        """Return log-space HBOS scores for all points."""
        X = np.column_stack([data[f] for f in self.feature_names])
        return np.array([self._compute_score(row) for row in X])
        
    def predict_outliers(self, data: defaultdict) -> np.ndarray:
        """
        Return boolean array where True indicates an outlier.
        """
        scores = self.decision_function(data)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return scores > threshold

class TypeNetFeatureExtractor:
    """
    Extract TypeNet keystroke features from raw keystroke data.
    Based on: Acien et al. (2021) TypeNet: Deep Learning Keystroke Biometrics
    """
    
    def __init__(self):
        self.error_types = {
            'valid': 'No error',
            'missing_key1_release': 'Missing key1 release',
            'missing_key1_press': 'Missing key1 press (orphan release)',
            'missing_key2_release': 'Missing key2 release',
            'missing_key2_press': 'Missing key2 press (orphan release)',
            'invalid_key1': 'Invalid key1 data',
            'invalid_key2': 'Invalid key2 data',
            'negative_timing': 'Negative timing value detected'
        }
    
    def parse_raw_file(self, filepath: str) -> pd.DataFrame:
        """
        Parse raw keystroke file with format: press-type (P or R), key, timestamp
        """
        try:
            # Read CSV without header
            df = pd.read_csv(filepath, header=None, names=['type', 'key', 'timestamp'])
            return df
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return pd.DataFrame()
    
    def match_press_release_pairs(self, df: pd.DataFrame) -> List[Dict]:
        """
        Match press and release events using a parentheses matching algorithm.
        Returns list of keystroke events with validity information.
        """
        events = []
        key_stacks = {}  # Stack for each unique key
        orphan_releases = []  # Store orphan release events
        
        for idx, row in df.iterrows():
            key = row['key']
            event_type = row['type']
            timestamp = row['timestamp']
            
            if key not in key_stacks:
                key_stacks[key] = deque()
            
            if event_type == 'P':
                # Push press event to stack
                key_stacks[key].append({
                    'key': key,
                    'press_time': timestamp,
                    'press_idx': idx
                })
            
            elif event_type == 'R':
                if key_stacks[key]:
                    # Match with most recent press
                    press_event = key_stacks[key].pop()
                    events.append({
                        'key': key,
                        'press_time': press_event['press_time'],
                        'release_time': timestamp,
                        'press_idx': press_event['press_idx'],
                        'release_idx': idx,
                        'valid': True,
                        'error': 'valid'
                    })
                else:
                    # Orphan release - no matching press
                    events.append({
                        'key': key,
                        'press_time': None,
                        'release_time': timestamp,
                        'press_idx': None,
                        'release_idx': idx,
                        'valid': False,
                        'error': 'missing_key1_press'
                    })
        
        # Handle unmatched press events (missing releases)
        for key, stack in key_stacks.items():
            while stack:
                press_event = stack.pop()
                events.append({
                    'key': press_event['key'],
                    'press_time': press_event['press_time'],
                    'release_time': None,
                    'press_idx': press_event['press_idx'],
                    'release_idx': None,
                    'valid': False,
                    'error': 'missing_key1_release'
                })
        
        # Sort events by press time (or release time for orphan releases)
        events.sort(key=lambda x: x['press_time'] if x['press_time'] is not None else x['release_time'])
        
        return events
    
    def calculate_features(self, key1_event: Dict, key2_event: Dict) -> Dict:
        """
        Calculate TypeNet features for a key pair.
        HL: Hold Latency (key1_release - key1_press)
        IL: Inter-key Latency (key2_press - key1_release)
        PL: Press Latency (key2_press - key1_press)
        RL: Release Latency (key2_release - key1_release)
        """
        features = {
            'key1': key1_event['key'],
            'key2': key2_event['key'],
            'key1_press': key1_event['press_time'],
            'key1_release': key1_event['release_time'],
            'key2_press': key2_event['press_time'],
            'key2_release': key2_event['release_time'],
            'HL': None,
            'IL': None,
            'PL': None,
            'RL': None,
            'valid': True,
            'error_description': 'No error'
        }
        
        # Check validity and set appropriate error messages
        if not key1_event['valid']:
            features['valid'] = False
            # Adjust error message for key1 position
            if key1_event['error'] == 'missing_key1_press':
                features['error_description'] = 'Missing key1 press (orphan release)'
            elif key1_event['error'] == 'missing_key1_release':
                features['error_description'] = 'Missing key1 release'
            else:
                features['error_description'] = self.error_types[key1_event['error']]
        
        if not key2_event['valid']:
            features['valid'] = False
            # Adjust error message for key2 position
            if key2_event['error'] == 'missing_key1_press':
                features['error_description'] = 'Missing key2 press (orphan release)'
            elif key2_event['error'] == 'missing_key1_release':
                features['error_description'] = 'Missing key2 release'
            else:
                features['error_description'] = self.error_types[key2_event['error']]
        
        # Calculate HL for key1 if it has both press and release (regardless of key2 validity)
        try:
            if key1_event['press_time'] is not None and key1_event['release_time'] is not None:
                features['HL'] = key1_event['release_time'] - key1_event['press_time']
                if features['HL'] < 0:
                    features['valid'] = False
                    features['error_description'] = 'Negative HL timing'
        except Exception as e:
            pass  # HL remains None if calculation fails
        
        # Calculate other features only if both keys are valid
        if key1_event['valid'] and key2_event['valid']:
            try:
                # IL: Inter-key Latency
                if key1_event['release_time'] is not None and key2_event['press_time'] is not None:
                    features['IL'] = key2_event['press_time'] - key1_event['release_time']
                    # IL can be negative (key overlap)
                
                # PL: Press Latency
                if key1_event['press_time'] is not None and key2_event['press_time'] is not None:
                    features['PL'] = key2_event['press_time'] - key1_event['press_time']
                    if features['PL'] < 0:
                        features['valid'] = False
                        features['error_description'] = 'Negative PL timing'
                
                # RL: Release Latency
                if key1_event['release_time'] is not None and key2_event['release_time'] is not None:
                    features['RL'] = key2_event['release_time'] - key1_event['release_time']
                    # RL can be negative (release order different from press order)
            
            except Exception as e:
                features['valid'] = False
                features['error_description'] = f'Calculation error: {str(e)}'
        
        return features
    
    def extract_features_from_file(self, filepath: str) -> pd.DataFrame:
        """
        Extract all features from a single raw keystroke file.
        """
        # Parse filename to get metadata
        filename = os.path.basename(filepath)
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) != 4:
            print(f"Invalid filename format: {filename}")
            return pd.DataFrame()
        
        platform_id, video_id, session_id, user_id = parts
        
        # Read and parse raw data
        raw_df = self.parse_raw_file(filepath)
        if raw_df.empty:
            return pd.DataFrame()
        
        # Match press-release pairs
        events = self.match_press_release_pairs(raw_df)
        
        # Extract features for consecutive key pairs
        features_list = []
        for i in range(len(events) - 1):
            key1_event = events[i]
            key2_event = events[i + 1]
            
            features = self.calculate_features(key1_event, key2_event)
            
            # Add metadata
            features.update({
                'user_id': int(user_id),
                'platform_id': int(platform_id),
                'video_id': int(video_id),
                'session_id': int(session_id),
                'sequence_id': i,
                'key1_timestamp': key1_event['press_time'] if key1_event['press_time'] is not None else key1_event['release_time']
            })
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply HBOS outlier detection to the feature data.
        Adds 'outlier' column to the dataframe.
        """
        # Only process valid records for outlier detection
        valid_df = df[df['valid']].copy()
        
        if len(valid_df) < 10:  # Too few records for meaningful outlier detection
            df['outlier'] = False
            return df
        
        # Prepare data for HBOS
        feature_data = defaultdict(list)
        timing_features = ['HL', 'IL', 'PL', 'RL']
        
        # Only use non-null timing features
        for feature in timing_features:
            valid_values = valid_df[feature].dropna()
            if len(valid_values) > 0:
                feature_data[feature] = valid_values.tolist()
        
        if not feature_data:  # No valid timing data
            df['outlier'] = False
            return df
        
        # Ensure all features have the same length by using indices
        valid_indices = valid_df.index
        aligned_data = defaultdict(list)
        
        for idx in valid_indices:
            has_all_features = True
            for feature in timing_features:
                if pd.notna(valid_df.loc[idx, feature]):
                    continue
                else:
                    has_all_features = False
                    break
            
            if has_all_features:
                for feature in timing_features:
                    aligned_data[feature].append(valid_df.loc[idx, feature])
        
        if not aligned_data or len(aligned_data[timing_features[0]]) < 10:
            df['outlier'] = False
            return df
        
        # Apply HBOS
        hbos = HBOS(n_bins=10, contamination=0.1, alpha=0.1, tol=0.5)
        hbos.fit(aligned_data)
        outliers = hbos.predict_outliers(aligned_data)
        
        # Initialize outlier column
        df['outlier'] = False
        
        # Map outliers back to original indices
        outlier_idx = 0
        for idx in valid_indices:
            has_all_features = True
            for feature in timing_features:
                if pd.isna(valid_df.loc[idx, feature]):
                    has_all_features = False
                    break
            
            if has_all_features:
                df.loc[idx, 'outlier'] = outliers[outlier_idx]
                outlier_idx += 1
        
        return df
    
    def process_user_directory(self, user_dir: str) -> pd.DataFrame:
        """
        Process all files for a single user.
        """
        all_features = []
        
        for filepath in Path(user_dir).glob('*.csv'):
            print(f"Processing {filepath}")
            features_df = self.extract_features_from_file(str(filepath))
            if not features_df.empty:
                all_features.append(features_df)
        
        if all_features:
            return pd.concat(all_features, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def process_dataset(self, root_dir: str, output_file: str = 'typenet_features_extracted.csv'):
        """
        Process entire dataset with one directory per user.
        """
        all_user_features = []
        
        # Process each user directory
        for user_dir in Path(root_dir).iterdir():
            if user_dir.is_dir():
                print(f"\nProcessing user directory: {user_dir}")
                user_features = self.process_user_directory(str(user_dir))
                if not user_features.empty:
                    # Apply outlier detection per user
                    user_features = self.detect_outliers(user_features)
                    all_user_features.append(user_features)
        
        # Combine all features
        if all_user_features:
            final_df = pd.concat(all_user_features, ignore_index=True)
            
            # Reorder columns to match expected format
            column_order = [
                'user_id', 'platform_id', 'video_id', 'session_id', 'sequence_id',
                'key1', 'key2', 'key1_press', 'key1_release', 'key2_press', 'key2_release',
                'HL', 'IL', 'PL', 'RL', 'key1_timestamp', 'valid', 'error_description', 'outlier'
            ]
            
            final_df = final_df[column_order]
            
            # Save to CSV
            final_df.to_csv(output_file, index=False)
            print(f"\nFeatures extracted and saved to {output_file}")
            print(f"Total records: {len(final_df)}")
            print(f"Valid records: {final_df['valid'].sum()}")
            print(f"Invalid records: {(~final_df['valid']).sum()}")
            print(f"Outliers detected: {final_df['outlier'].sum()}")
            
            # Print error summary
            print("\nError summary:")
            error_counts = final_df[~final_df['valid']]['error_description'].value_counts()
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
            
            # Print outlier summary
            print("\nOutlier summary:")
            outlier_valid = final_df[final_df['valid'] & final_df['outlier']]
            print(f"  Outliers in valid data: {len(outlier_valid)}")
            print(f"  Outlier rate in valid data: {len(outlier_valid) / final_df['valid'].sum() * 100:.2f}%")
            
            return final_df
        else:
            print("No features extracted from dataset")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = TypeNetFeatureExtractor()
    
    print("=== TypeNet Feature Extraction with Outlier Detection ===\n")
    
    # Process the demo dataset
    print("Processing demo dataset...")
    raw_data_dir = 'data_dump/loadable_Combined_HU_HT'
    
    # Make saved processed data fileanames consistent for automated download and extraction.
    hostname = socket.gethostname()
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_save_dir = os.path.join(base_dir, f"processed_data-{now}-{hostname}")
    os.makedirs(processed_data_save_dir, exist_ok=True)
    save_data_path = os.path.join(processed_data_save_dir,'typenet_features.csv')

    features_df = extractor.process_dataset(raw_data_dir, save_data_path)
    
    print("\n=== Extracted Features ===")
    print(features_df.head(20).to_string(index=False))
    
    print("\n=== Feature Statistics ===")
    print(f"Total keypair features: {len(features_df)}")
    print(f"Valid features: {features_df['valid'].sum()}")
    print(f"Invalid features: {(~features_df['valid']).sum()}")
    print(f"Outliers: {features_df['outlier'].sum()}")
    
    print("\n=== Timing Feature Ranges (valid records only) ===")
    valid_df = features_df[features_df['valid']]
    if not valid_df.empty:
        for feature in ['HL', 'IL', 'PL', 'RL']:
            if feature in valid_df.columns and valid_df[feature].notna().any():
                print(f"{feature}: min={valid_df[feature].min():.0f}ms, "
                      f"max={valid_df[feature].max():.0f}ms, "
                      f"mean={valid_df[feature].mean():.0f}ms")
    
    print("\n=== Error Analysis ===")
    error_df = features_df[~features_df['valid']]
    if not error_df.empty:
        print("Errors found:")
        for error_type, count in error_df['error_description'].value_counts().items():
            print(f"  - {error_type}: {count} occurrences")
    
    print("\n=== Outlier Analysis ===")
    outlier_df = features_df[features_df['outlier']]
    if not outlier_df.empty:
        print("Outliers by validity:")
        print(f"  - Valid outliers: {outlier_df['valid'].sum()}")
        print(f"  - Invalid outliers: {(~outlier_df['valid']).sum()}")
        
        print("\nOutliers by platform:")
        for platform in sorted(features_df['platform_id'].unique()):
            platform_outliers = outlier_df[outlier_df['platform_id'] == platform]
            print(f"  - Platform {platform}: {len(platform_outliers)} outliers")