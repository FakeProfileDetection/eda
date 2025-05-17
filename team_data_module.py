"""
team_data.py - Helper module for accessing team data
"""

import os
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
import glob

class TeamData:
    """Helper class for accessing the team's dataset."""
    
    def __init__(self):
        """Initialize the TeamData helper."""
        # Get data path from environment variable
        self.data_path = os.environ.get('DATA_PATH')
        if not self.data_path:
            raise EnvironmentError(
                "DATA_PATH environment variable not set. "
                "Please run setup.py first."
            )
        
        self.data_dir = Path(self.data_path)
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory '{self.data_dir}' not found. "
                "Please run setup.py to download the data."
            )
    
    def get_user_ids(self) -> List[int]:
        """Get a list of all available user IDs."""
        user_dirs = self.data_dir.glob("user-*")
        user_ids = []
        
        for user_dir in user_dirs:
            try:
                # Extract user ID from directory name
                user_id = int(user_dir.name.split("-")[1])
                user_ids.append(user_id)
            except (IndexError, ValueError):
                continue
        
        return sorted(user_ids)
    
    def get_files_for_user(self, user_id: int) -> List[str]:
        """Get list of available data files for a specific user."""
        user_dir = self.data_dir / f"user-{user_id}"
        if not user_dir.exists():
            raise FileNotFoundError(f"No data directory found for user-{user_id}")
        
        return [f.name for f in user_dir.glob("*.csv")]
    
    def load_user_file(self, user_id: int, file_name: str) -> pd.DataFrame:
        """Load a specific data file for a user."""
        file_path = self.data_dir / f"user-{user_id}" / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_name} not found for user-{user_id}")
        
        return pd.read_csv(file_path)
    
    def load_all_user_data(self, user_id: int) -> Dict[str, pd.DataFrame]:
        """Load all data files for a specific user."""
        files = self.get_files_for_user(user_id)
        data_dict = {}
        
        for file_name in files:
            # Use the filename without extension as the key
            key = Path(file_name).stem
            data_dict[key] = self.load_user_file(user_id, file_name)
            
        return data_dict
    
    def load_specific_file_for_all_users(self, file_name: str) -> Dict[int, pd.DataFrame]:
        """Load a specific file across all users."""
        user_ids = self.get_user_ids()
        data_dict = {}
        
        for user_id in user_ids:
            try:
                data_dict[user_id] = self.load_user_file(user_id, file_name)
            except FileNotFoundError:
                continue
            
        return data_dict
    
    def search_data(self, pattern: str) -> List[Path]:
        """Search for files matching a pattern in the data directory."""
        search_path = self.data_dir / "**" / pattern
        return list(Path(self.data_path).glob(str(search_path.relative_to(self.data_path))))


# Create a singleton instance for easy import
data = TeamData()


# Example usage
if __name__ == "__main__":
    # Get all user IDs
    print(f"Available user IDs: {data.get_user_ids()}")
    
    # Get files for user 1
    print(f"Files for user-1: {data.get_files_for_user(1)}")
    
    # Load data1.csv for user 1
    df = data.load_user_file(1, "data1.csv")
    print(f"Preview of user-1's data1.csv:")
    print(df.head())
    
    # Search for all data1 files
    data1_files = data.search_data("data1.csv")
    print(f"Found {len(data1_files)} data1.csv files")
