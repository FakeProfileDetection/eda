"""process_raw_data.py

AUTHOR: Lori Pickering, sol@solofai.com
DATE: 18 MAY 2025
DESCRIPTION: This script processes the raw data from the data_sump directory and 
saves the processed data to the processed_data directory, which named
by the datetime-hostname of the machine running the script. 

Data is saved as a csv file.
"""

import os
import polars as pl
import pandas as pd
from typing import List, Dict, Tuple, Union, Callable, Any, Optional
import multiprocessing
from tqdm import tqdm
from datetime import datetime


valid_platforms = [1,2,3]
valid_sessions = [1,2]
valid_videos = [1,2,3]

def process_raw_file_typenet_features_polars(args: Tuple[int, str]) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    user_id, filepath = args
    filename = os.path.basename(filepath)

    try:
        # Use Polars to read and for initial column setup
        df = pl.read_csv(
            filepath,
            has_header=False,
            new_columns=["press_type", "key", "timestamp_orig"],
            dtypes={
                "press_type": pl.Categorical, # More efficient for P/R
                "key": pl.Utf8,
                "timestamp_orig": pl.Int64
            }
        )
    except Exception as e:
        # Log critical load error, maybe return empty/None
        print(f"Could not load dataframe from {filepath}: {e}")
        return None, pl.DataFrame({ # Return empty log with error
            "user_id": [user_id], "filename": [filename], "sequence_id": [0],
            "key1": [None], "key2": [None], "key1_index": [None], "key2_index": [None],
            "break_message": [f"CSV load error: {e}"]
        })

    if df.is_empty():
        return None, None # Or an empty log entry

    df = df.with_columns(
        pl.col("key").str.replace_all("'", "").str.replace_all('"', "")
    )
    
    # Sort by timestamp and add original index, normalize timestamp
    df = df.sort("timestamp_orig").with_row_count("original_idx")
    if df["timestamp_orig"].min() is None: # Empty after filter or all nulls
         return None, None
    df = df.with_columns(
        (pl.col("timestamp_orig") - df["timestamp_orig"].min()).alias("timestamp")
    )

    # --- Log generation setup ---
    log_breaks_list = []

    # --- Identify valid P-R pairs for individual keys ---
    # For each key, find its next event
    df_with_next_for_key = df.with_columns(
        df.sort(["key", "timestamp"]).select([
            pl.col("timestamp").shift(-1).over("key").alias("next_ts_for_key"),
            pl.col("press_type").shift(-1).over("key").alias("next_pt_for_key"),
            pl.col("original_idx").shift(-1).over("key").alias("next_idx_for_key"),
        ])
    )

    # Log presses that are not properly released before another event of the same key
    # or end of file for that key
    broken_presses = df_with_next_for_key.filter(
        (pl.col("press_type") == "P") &
        ((pl.col("next_pt_for_key") != "R") | pl.col("next_pt_for_key").is_null())
    )
    if not broken_presses.is_empty():
        for row in broken_presses.iter_rows(named=True):
            log_breaks_list.append({
                "user_id": user_id, "filename": filename, "sequence_id": 0, # Simplified sequence_id for logs
                "key1": row["key"], "key2": None, 
                "key1_index": row["original_idx"], "key2_index": None,
                "break_message": "key1_release_not_found_before_next_press_or_eof"
            })

    # Valid single P-R pairs
    df_single_pr = df_with_next_for_key.filter(
        (pl.col("press_type") == "P") & (pl.col("next_pt_for_key") == "R")
    ).select([
        pl.col("key").alias("k_key"),
        pl.col("timestamp").alias("k_press_ts"),
        pl.col("next_ts_for_key").alias("k_release_ts"),
        pl.col("original_idx").alias("k_press_idx"),
        pl.col("next_idx_for_key").alias("k_release_idx"),
    ]).sort("k_press_ts") # Critically sort by press time for sequential pairing

    if df_single_pr.is_empty():
        df_log_breaks = pl.DataFrame(log_breaks_list) if log_breaks_list else None
        return None, df_log_breaks

    # --- Form P1R1 and P2R2 sequences ---
    # df_k1 is current P-R pair, df_k2 is the next P-R pair in the timeline
    df_k1 = df_single_pr.rename({
        "k_key": "key1", "k_press_ts": "k1_press_ts", "k_release_ts": "k1_release_ts",
        "k_press_idx": "k1_press_idx", "k_release_idx": "k1_release_idx"
    })
    df_k2 = df_single_pr.shift(-1).rename({
        "k_key": "key2", "k_press_ts": "k2_press_ts", "k_release_ts": "k2_release_ts",
        "k_press_idx": "k2_press_idx", "k_release_idx": "k2_release_idx"
    })

    # Combine k1 and k2 data. Use unique suffixes if any columns were not renamed.
    # Polars automatically handles this if df_k1 and df_k2 have distinct column names.
    features_candidates = pl.concat([df_k1, df_k2], how="horizontal").drop_nulls("key2") # key2 is null for the last shifted element

    # Filter for valid TypeNet sequences: P2 must occur after R1
    valid_features_df = features_candidates.filter(pl.col("k2_press_ts") > pl.col("k1_release_ts"))
    
    # Log P1R1 pairs that don't form a valid TypeNet sequence with the *next* P-R pair
    # These are P1R1 where (k2_press_ts <= k1_release_ts) or where k2 is null (end of sequence)
    broken_digraphs = features_candidates.filter(
        (pl.col("k2_press_ts") <= pl.col("k1_release_ts")) | pl.col("key2").is_null()
    )
    # Also, the very last PR pair in df_single_pr if it wasn't part of features_candidates
    # This logging needs to be more precise to match old script's sequence_id logic if necessary.
    # For now, this is a simpler log of unusable P1R1 pairs for forming digraphs.
    if not broken_digraphs.is_empty():
        for row in broken_digraphs.iter_rows(named=True):
            log_breaks_list.append({
                "user_id": user_id, "filename": filename, "sequence_id": 0,
                "key1": row["key1"], "key2": row.get("key2"), # key2 might be null
                "key1_index": row["k1_press_idx"], 
                "key2_index": row.get("k2_press_idx"), # k2_press_idx might be null
                "break_message": "p1r1_not_forming_valid_typenet_digraph_with_next_pr"
            })


    if valid_features_df.is_empty():
        df_log_breaks = pl.DataFrame(log_breaks_list) if log_breaks_list else None
        return None, df_log_breaks

    # Assign metadata and calculate TypeNet features
    platform_id, video_id, session_id, _ = filename.replace(".csv", "").split("_")
    
    final_df = valid_features_df.with_columns([
        pl.lit(user_id).alias("user_id"),
        pl.lit(int(platform_id)).alias("platform_id"),
        pl.lit(int(video_id)).alias("video_id"),
        pl.lit(int(session_id)).alias("session_id"),
        pl.lit(0).alias("sequence_id"), # Simplified sequence_id
        (pl.col("k1_release_ts") - pl.col("k1_press_ts")).alias("HL"),
        (pl.col("k2_press_ts") - pl.col("k1_release_ts")).alias("IL"),
        (pl.col("k2_press_ts") - pl.col("k1_press_ts")).alias("PL"),
        (pl.col("k2_release_ts") - pl.col("k1_release_ts")).alias("RL"),
        pl.col("k1_press_ts").alias("key1_timestamp") # Original name for this timestamp
    ]).select([ # Select and order columns as expected
        "user_id", "platform_id", "video_id", "session_id", "sequence_id",
        "key1", "key2", 
        "HL", "IL", "PL", "RL", 
        "key1_timestamp",
        # Optional: include original indices or timestamps for debugging if needed
        # "k1_press_ts", "k1_release_ts", "k2_press_ts", "k2_release_ts",
        # "k1_press_idx", "k1_release_idx", "k2_press_idx", "k2_release_idx"
    ])
    
    df_log_breaks = pl.DataFrame(log_breaks_list) if log_breaks_list else None
    if df_log_breaks is not None and df_log_breaks.is_empty(): # Ensure None if truly empty list
        df_log_breaks = None

    return final_df, df_log_breaks
        

class ExtractTypeNetFeatures:
    def __init__(self, raw_dir: str, 
                 verbose: bool = True, 
                 debug: bool = False,
                 use_multiprocessing: bool = True,
                 num_workers: int = None, 
                 zero_timestamp: bool=True) -> None:
        """Extract TypeNet features from raw data files.
        
        Args:
            raw_dir (str): Path to the directory containing user subdirectories.
            verbose (bool): If True, print diagnostic messages.
            debug (bool): If True, run in debug mode.
            use_multiprocessing (bool): If True, process files in parallel.
            num_workers (int): Number of worker processes (if using multiprocessing). Defaults to os.cpu_count().
        """
        self.raw_dir = raw_dir
        self.debug = debug
        self.verbose = verbose
        self.raw_dir = raw_dir
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.zero_timestamp = zero_timestamp

        if not os.path.exists(raw_dir):
            raise FileNotFoundError(f"Directory not found: {raw_dir}")
        
        try:
            self.user_files, self.bad_users = self._get_user_files(raw_dir)
            self.data, self.log_breaks = self._get_typeNet_data()  # DataFrame with features and metadata
        except Exception as e:
            try:
                print(f"Error extracting data: {self.data}")
            except Exception as e2:
                print(f"Error extracting data: {e2}")
            raise e
        
    def save_data(self, save_dir: str) -> None:
        """Save the extracted data to a file."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        save_path = os.path.join(save_dir, "typenet_features.csv")
        try:
            self.data.write_csv(save_path, include_header=True)
        except Exception as e:
            print(f"Error saving data: {e}")
        
        log_path = os.path.join(save_dir, "typenet_log_breaks.csv")
        if self.log_breaks is not None:
            try:
                self.log_breaks.write_csv(log_path, include_header=True)
            except Exception as e:
                print(f"Error saving log breaks: {e}")
        
        if self.verbose:
            print(f"Saved data to {save_path}")
            print(f"Saved log breaks to {log_path}")
            
    def _get_user_files(self, raw_dir: str) -> Tuple[Dict, List]:
        user_dirs = {
                d: os.path.join(raw_dir, d) 
                for d in sorted(os.listdir(raw_dir)) 
                if os.path.isdir(os.path.join(raw_dir, d))
            }
        if self.verbose:
            print(f"Found {len(user_dirs.keys())} user directories.")
            
        ### Verify all the files are present
        user_files = {}
        bad_users = []
        for user_id_str, dirpath in user_dirs.items():
            try:
                user_id = int(user_id_str)
            except ValueError:
                if self.verbose:
                    print(f"Skipping directory {user_id_str} (cannot convert to int)")
                continue
                
            user_files[user_id] = []
            files = [f for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
            
            # Filter files that are not in the correct format
            for platform in valid_platforms:
                for session in valid_sessions:
                    for video in valid_videos:
                        filename = f"{platform}_{video}_{session}_{user_id}.csv"
                        if filename in files:
                            user_files[user_id].append(os.path.join(dirpath,  filename))
                            
            # Verify all files are present--filter data with missing files
            if len(user_files[user_id]) != 18:
                bad_users.append(user_id)
                del user_files[user_id]
            
        if self.verbose:
            print(f"Found {len(user_files.keys())} users with complete data.\n{user_files.keys()}")
            print(f"Found {len(bad_users)} users with incomplete data.\n{bad_users}")
            
        return user_files, bad_users
        
    def _get_typeNet_data(self) -> pl.DataFrame:
        """
        Process all files (across users) and combine their extracted features into a single DataFrame.
        """
        all_args = []  # List of tuples: (user_id, filepath, sequence_gap_threshold, ignore_modifiers)
        for user_id, files in self.user_files.items():
            for filepath in files:
                all_args.append((user_id, filepath))
                
        dataframes = []
        errors  = []
        multiprocessing.set_start_method('forkserver', force=True)
        if self.num_workers is not None:
            cores = multiprocessing.cpu_count() # TODO: I'm not using this.  Should this be switched with num_workers?
        else:
            cores = self.num_workers
        
        if self.use_multiprocessing:
            if self.verbose:
                print(f"Processing {len(all_args)} files with {self.num_workers} workers...")
                
            with multiprocessing.Pool(self.num_workers) as pool:
            # with multiprocessing.Pool(cores) as pool:
                for results in tqdm(pool.imap_unordered(process_raw_file_typenet_features, all_args), total=len(all_args)):
                    df_data, df_errors = results
                    if not df_data.is_empty():
                        dataframes.append(df_data)
                    if not df_errors.is_empty():
                        errors.append(df_errors)        
        else:
            # combined_df = pl.DataFrame() # Not needed here, concat list of dfs at the end
            # log_breaks_intermediate_list = [] # Use this for errors
            if self.verbose:
                print(f"Processing {len(all_args)} files sequentially...")
            for idx, args in enumerate(tqdm(all_args, total=len(all_args))):
                try:
                    # Use the new polars-based function
                    df_data, df_errors = process_raw_file_typenet_features_polars(args) # MODIFIED
                    
                    if df_data is not None and not df_data.is_empty(): # MODIFIED
                        dataframes.append(df_data)
                        
                    if df_errors is not None and not df_errors.is_empty(): # MODIFIED
                        # Ensure correct dtypes if concatenating later, or if schema might vary
                        # This might not be necessary if process_raw_file_typenet_features_polars
                        # is consistent, but good for robustness if schema issues arise.
                        # df_errors = df_errors.with_columns(
                        #     pl.col("key2_index").cast(pl.Int64, strict=False), # strict=False if column might be all null
                        #     pl.col("key1_index").cast(pl.Int64, strict=False),
                        #     pl.col("sequence_id").cast(pl.Int64, strict=False)
                        # )
                        errors.append(df_errors) # Append to list, concat later
                # ... (rest of try-except) ...
            
            # Concatenate all collected dataframes and errors once
            if not dataframes:
                 # Handle case with absolutely no data, e.g. raise error or return empty structure
                 print("Warning: No data was extracted from any file.")
                 # Depending on desired behavior, you might want to create an empty df with schema
                 # For now, assuming downstream code can handle empty combined_df if dataframes list is empty
            combined_df = pl.concat(dataframes) if dataframes else pl.DataFrame() # Create empty if list is empty
            
            if not errors:
                log_breaks = None # Or pl.DataFrame() with schema if preferred for consistency
            else:
                try:
                    log_breaks = pl.concat(errors)
                except Exception as e:
                    print(f"Error concatenating log_breaks: {e}")
                    # Handle error, possibly print individual error dfs
                    for i, err_df in enumerate(errors):
                        print(f"Error df {i}:")
                        with pl.Config(tbl_rows=5, tbl_cols=-1):
                             print(err_df)
                             print(err_df.schema)
                    raise e
            
            # This return was outside the else block, should be for the whole function
            # return combined_df, log_breaks 
        
        # This section should be OUTSIDE the if self.use_multiprocessing / else block
        # It processes the results from EITHER path

        if not dataframes:
            print("No data extracted from any files.")
            # Return empty DataFrames with expected schema or raise error
            # For example:
            # schema_features = {"user_id": pl.Int64, ...} # define expected schema
            # combined_df = pl.DataFrame(schema=schema_features)
            # schema_logs = {"user_id": pl.Int64, ...}
            # log_breaks_final = pl.DataFrame(schema=schema_logs)
            # return combined_df, log_breaks_final
            raise ValueError("No data extracted from any files after processing.") # Or handle as per requirements


        print(f"Number of data file segments extracted: {len(dataframes)}")
        combined_df_final = pl.concat(dataframes)
        print("Combined features DataFrame:")
        with pl.Config(tbl_rows=5, tbl_cols=-1): # Print head
            print(combined_df_final.head())
            
        log_breaks_final = None
        if errors:
            print(f"Number of error log segments: {len(errors)}")
            try:
                log_breaks_final = pl.concat(errors)
                print("Combined log_breaks DataFrame:")
                with pl.Config(tbl_rows=5, tbl_cols=-1): # Print head
                    print(log_breaks_final.head())
            except Exception as e:
                print(f"Critical error concatenating final log_breaks: {e}")
                # Potentially save individual error logs or re-raise
                # For now, we'll let it be None if concat fails here
                log_breaks_final = None # Or some indicator of failure
        else:
            print("No error logs generated.")
            
        return combined_df_final, log_breaks_final

# In main() or at the top of the script:
if __name__ == "__main__":
    # Set multiprocessing start method here, once.
    # forkserver is good on Linux/macOS if available. 'spawn' is safer default for macOS/Windows.
    try:
        multiprocessing.set_start_method('spawn', force=True) # or 'forkserver'
    except RuntimeError:
        print("Note: Multiprocessing start method already set or OS does not support changing it here.")
        pass # Or handle as appropriate for your environment

    main()