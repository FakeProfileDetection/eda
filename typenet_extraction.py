import pandas as pd
import numpy as np
import os
from pathlib import Path
from collections import deque
from typing import List, Tuple, Dict, Optional
import socket
from datetime import datetime

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
                    all_user_features.append(user_features)
        
        # Combine all features
        if all_user_features:
            final_df = pd.concat(all_user_features, ignore_index=True)
            
            # Reorder columns to match expected format
            column_order = [
                'user_id', 'platform_id', 'video_id', 'session_id', 'sequence_id',
                'key1', 'key2', 'key1_press', 'key1_release', 'key2_press', 'key2_release',
                'HL', 'IL', 'PL', 'RL', 'key1_timestamp', 'valid', 'error_description'
            ]
            
            final_df = final_df[column_order]
            
            # Save to CSV
            final_df.to_csv(output_file, index=False)
            print(f"\nFeatures extracted and saved to {output_file}")
            print(f"Total records: {len(final_df)}")
            print(f"Valid records: {final_df['valid'].sum()}")
            print(f"Invalid records: {(~final_df['valid']).sum()}")
            
            # Print error summary
            print("\nError summary:")
            error_counts = final_df[~final_df['valid']]['error_description'].value_counts()
            for error, count in error_counts.items():
                print(f"  {error}: {count}")
            
            return final_df
        else:
            print("No features extracted from dataset")
            return pd.DataFrame()


# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = TypeNetFeatureExtractor()
    
    # Example: Process a single file
    # df = extractor.extract_features_from_file('path/to/1_1_1_1001.csv')
    
    # Example: Process entire dataset
    # df = extractor.process_dataset('path/to/dataset/root')
    
    # Example: Process with custom output filename
    # df = extractor.process_dataset('path/to/dataset/root', 'my_features.csv')
    
    # print("TypeNet Feature Extractor ready to use!")
    # print("\nUsage examples:")
    # print("1. Process single file:")
    # print("   df = extractor.extract_features_from_file('path/to/1_1_1_1001.csv')")
    # print("\n2. Process entire dataset:")
    # print("   df = extractor.process_dataset('path/to/dataset/root')")
    
    print("=== TypeNet Feature Extraction  ===\n")
    
    # Process the demo dataset
    print("Processing demo dataset...")
    raw_data_dir = 'data_dump/loadable_Combined_HU_HT'
    
    # Make saved processed data fileanames consistent for automated download and extraction.
    hostname = socket.gethostname()
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S") # Removed Min and Sec suffix for brevity
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Or use os.getcwd() if script is run from its location
    processed_data_save_dir = os.path.join(base_dir, f"processed_data-{now}-{hostname}")
    os.makedirs(processed_data_save_dir, exist_ok=True)
    save_data_path = os.path.join(processed_data_save_dir,'typenet_features.csv')

    features_df = extractor.process_dataset(raw_data_dir, save_data_path)
    
    print("\n=== Extracted Features ===")
    print(features_df.to_string(index=False))
    
    print("\n=== Feature Statistics ===")
    print(f"Total keypair features: {len(features_df)}")
    print(f"Valid features: {features_df['valid'].sum()}")
    print(f"Invalid features: {(~features_df['valid']).sum()}")
    
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
        
        print("\nExample error records:")
        print(error_df[['key1', 'key2', 'error_description']].head(5).to_string(index=False))
    else:
        print("No errors found in the dataset!")
    
    # Cleanup
    # import shutil
    # shutil.rmtree('demo_data')
    # os.remove('demo_features.csv')
    
    print(features_df)