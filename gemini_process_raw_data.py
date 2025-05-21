"""
process_keystroke_data.py

DESCRIPTION: This script processes raw keystroke data to extract TypeNet features.
             It reads CSV files from user-specific subdirectories, processes them
             in parallel, and saves the combined features and processing logs
             to new CSV files.
"""

import os
import polars as pl
from typing import List, Tuple, Optional, Dict, Any # Added Any and Dict
import multiprocessing
from tqdm import tqdm
from datetime import datetime
import socket # For hostname

# --- Configuration ---
# These were global in the original script. Define them clearly.
VALID_PLATFORMS = [1, 2, 3]
VALID_SESSIONS = [1, 2]
VALID_VIDEOS = [1, 2, 3]
EXPECTED_FILES_PER_USER = len(VALID_PLATFORMS) * len(VALID_SESSIONS) * len(VALID_VIDEOS)

# Define schemas for consistency, especially for logs and empty DataFrames
LOG_SCHEMA = {
    "user_id": pl.Int64,
    "filename": pl.Utf8,
    "sequence_id": pl.Int64,
    "key1": pl.Utf8,
    "key2": pl.Utf8,
    "key1_index": pl.Int64, # Refers to original_idx in the source CSV
    "key2_index": pl.Int64, # Refers to original_idx in the source CSV
    "break_message": pl.Utf8
}

FEATURE_SCHEMA = {
    "user_id": pl.Int64,
    "platform_id": pl.Int32, # Changed to Int32 as they are small integers
    "video_id": pl.Int32,
    "session_id": pl.Int32,
    "sequence_id": pl.Int64,
    "key1": pl.Utf8,
    "key2": pl.Utf8,
    "HL": pl.Int64, # Hold Latency (duration)
    "IL": pl.Int64, # Inter-key Latency (duration)
    "PL": pl.Int64, # Press Latency (duration)
    "RL": pl.Int64, # Release Latency (duration)
    "key1_timestamp": pl.Int64, # Normalized timestamp of key1 press
    # Storing original indices can be useful for debugging/tracing back
    "k1_press_idx_original": pl.UInt32, # Using UInt32 for row counts
    "k1_release_idx_original": pl.UInt32,
    "k2_press_idx_original": pl.UInt32,
    "k2_release_idx_original": pl.UInt32,
}


def process_single_file_polars(args: Tuple[int, str]) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """
    Processes a single raw keystroke CSV file to extract TypeNet features and log breaks.

    Args:
        args: A tuple containing (user_id, filepath).

    Returns:
        A tuple containing (features_df, logs_df).
        features_df: Polars DataFrame with TypeNet features.
        logs_df: Polars DataFrame with logged processing breaks/errors.
        Returns (None, logs_df) or (None, None) on critical errors or no data.
    """
    user_id, filepath = args
    filename = os.path.basename(filepath)
    log_breaks_list: List[Dict[str, Any]] = []

    try:
        df = pl.read_csv(
            filepath,
            has_header=False,
            new_columns=["press_type", "key", "timestamp_orig"],
            dtypes={
                "press_type": pl.Categorical,
                "key": pl.Utf8,
                "timestamp_orig": pl.Int64
            }
        )
    except Exception as e:
        error_message = f"CSV load error: {e}"
        print(f"Error processing {filename} for user {user_id}: {error_message}")
        log_entry = {
            "user_id": user_id, "filename": filename, "sequence_id": 0,
            "key1": None, "key2": None, "key1_index": None, "key2_index": None,
            "break_message": error_message
        }
        return None, pl.DataFrame([log_entry], schema=LOG_SCHEMA)

    if df.is_empty():
        return None, None # No features, no specific log entry needed beyond potential file discovery logs

    # Preprocessing
    df = df.with_columns(
        pl.col("key").str.replace_all("'", "").str.replace_all('"', "")
    ).sort("timestamp_orig").with_row_count("original_idx")

    min_timestamp = df["timestamp_orig"].min()
    if min_timestamp is None: # Should not happen if df is not empty, but good check
        return None, None
    df = df.with_columns(
        (pl.col("timestamp_orig") - min_timestamp).alias("timestamp")
    )

    # --- Step 1: Identify valid atomic Press-Release (P-R) pairs for individual keys ---
    # A P-R pair is valid if a 'P' is followed by an 'R' for the SAME key,
    # with no other 'P' of that SAME key in between.
    df_with_next_for_key = df.with_columns(
        # These columns look at the *next* event *for the same key*
        pl.col("timestamp").shift(-1).over("key").alias("next_ts_for_key"),
        pl.col("press_type").shift(-1).over("key").alias("next_pt_for_key"),
        pl.col("original_idx").shift(-1).over("key").alias("next_idx_for_key")
    )

    # Log presses that are not validly released
    broken_presses = df_with_next_for_key.filter(
        (pl.col("press_type") == "P") &
        ((pl.col("next_pt_for_key") != "R") | pl.col("next_pt_for_key").is_null())
    )
    for row in broken_presses.iter_rows(named=True):
        log_breaks_list.append({
            "user_id": user_id, "filename": filename, "sequence_id": 0, # Placeholder sequence_id for this type of log
            "key1": row["key"], "key2": None,
            "key1_index": row["original_idx"], "key2_index": None,
            "break_message": "key1_release_not_found_or_interrupted_by_same_key_press"
        })

    # Create DataFrame of all valid single P-R pairs, sorted by their press time
    df_single_pr = df_with_next_for_key.filter(
        (pl.col("press_type") == "P") & (pl.col("next_pt_for_key") == "R")
    ).select([
        pl.col("key").alias("k_key"),
        pl.col("timestamp").alias("k_press_ts"),
        pl.col("next_ts_for_key").alias("k_release_ts"),
        pl.col("original_idx").alias("k_press_idx"), # original_idx of the Press event
        pl.col("next_idx_for_key").alias("k_release_idx") # original_idx of the Release event
    ]).sort("k_press_ts")

    if df_single_pr.is_empty():
        log_df = pl.DataFrame(log_breaks_list, schema_overrides=LOG_SCHEMA) if log_breaks_list else None
        if log_df is not None and log_df.is_empty(): log_df = None # Ensure None if truly empty
        return None, log_df

    # --- Step 2: Form P1R1-P2R2 feature candidates ---
    # df_k1 represents the first P-R pair in a digraph (Key1 Press, Key1 Release)
    df_k1 = df_single_pr.rename({
        "k_key": "key1", "k_press_ts": "k1_press_ts", "k_release_ts": "k1_release_ts",
        "k_press_idx": "k1_press_idx_original", "k_release_idx": "k1_release_idx_original"
    })
    # df_k2 represents the second P-R pair (Key2 Press, Key2 Release),
    # taken as the *next* P-R pair from df_single_pr's timeline.
    df_k2 = df_single_pr.shift(-1).rename({
        "k_key": "key2", "k_press_ts": "k2_press_ts", "k_release_ts": "k2_release_ts",
        "k_press_idx": "k2_press_idx_original", "k_release_idx": "k2_release_idx_original"
    })

    # Combine to form candidates: (P1R1 event) followed by (P2R2 event)
    # Each row is a candidate for a TypeNet feature set.
    # drop_nulls("key2") removes the last P-R pair which cannot start a P1R1-P2R2 sequence.
    feature_candidates = pl.concat([df_k1, df_k2], how="horizontal").drop_nulls("key2")

    if feature_candidates.is_empty():
        # This means there were fewer than two valid P-R pairs in the file.
        # Log the remaining P-R pairs in df_single_pr as they didn't form digraphs.
        for row in df_single_pr.iter_rows(named=True):
             log_breaks_list.append({
                "user_id": user_id, "filename": filename, "sequence_id": 0,
                "key1": row["k_key"], "key2": None,
                "key1_index": row["k_press_idx"], "key2_index": None,
                "break_message": "single_pr_pair_did_not_form_digraph"
            })
        log_df = pl.DataFrame(log_breaks_list, schema_overrides=LOG_SCHEMA) if log_breaks_list else None
        if log_df is not None and log_df.is_empty(): log_df = None
        return None, log_df

    # --- Step 3: Calculate sequence_id based on breaks in contiguity ---
    # A break occurs if a candidate P1R1-P2R2 is not a valid TypeNet digraph
    # (i.e., P2 press is not after P1 release).
    feature_candidates = feature_candidates.with_columns(
        (pl.col("k2_press_ts") > pl.col("k1_release_ts")).alias("is_valid_digraph")
    )
    # `is_break_for_sequence` is true if the current candidate is NOT a valid digraph.
    # The `cumsum` over these breaks generates the sequence_id.
    # Each time a break occurs, the sequence_id increments.
    feature_candidates = feature_candidates.with_columns(
        (~pl.col("is_valid_digraph")).cum_sum().alias("sequence_id")
    )

    # Log the P1R1-P2R2 candidates that are *not* valid TypeNet digraphs
    invalid_digraph_candidates = feature_candidates.filter(~pl.col("is_valid_digraph"))
    for row in invalid_digraph_candidates.iter_rows(named=True):
        log_breaks_list.append({
            "user_id": user_id, "filename": filename, "sequence_id": row["sequence_id"],
            "key1": row["key1"], "key2": row["key2"],
            "key1_index": row["k1_press_idx_original"],
            "key2_index": row["k2_press_idx_original"],
            "break_message": "p2_press_not_after_p1_release"
        })

    # --- Step 4: Filter for valid TypeNet features ---
    # Only keep the candidates that form valid digraphs.
    valid_features_df = feature_candidates.filter(pl.col("is_valid_digraph"))

    if valid_features_df.is_empty():
        log_df = pl.DataFrame(log_breaks_list, schema_overrides=LOG_SCHEMA) if log_breaks_list else None
        if log_df is not None and log_df.is_empty(): log_df = None
        return None, log_df

    # --- Step 5: Assign metadata and calculate TypeNet feature values ---
    try:
        platform_id_str, video_id_str, session_id_str, _ = filename.replace(".csv", "").split("_")
        platform_id, video_id, session_id = int(platform_id_str), int(video_id_str), int(session_id_str)
    except ValueError as e:
        error_message = f"Invalid filename format for metadata: {filename} ({e})"
        print(f"Error processing {filename} for user {user_id}: {error_message}")
        # Log this critical error for the file
        log_breaks_list.append({
            "user_id": user_id, "filename": filename, "sequence_id": 0,
            "key1": None, "key2": None, "key1_index": None, "key2_index": None,
            "break_message": error_message
        })
        log_df = pl.DataFrame(log_breaks_list, schema_overrides=LOG_SCHEMA) if log_breaks_list else None
        if log_df is not None and log_df.is_empty(): log_df = None
        return None, log_df # No features can be reliably attributed if metadata fails

    final_features_df = valid_features_df.with_columns([
        pl.lit(user_id).cast(pl.Int64).alias("user_id"),
        pl.lit(platform_id).cast(pl.Int32).alias("platform_id"),
        pl.lit(video_id).cast(pl.Int32).alias("video_id"),
        pl.lit(session_id).cast(pl.Int32).alias("session_id"),
        # sequence_id is already calculated
        (pl.col("k1_release_ts") - pl.col("k1_press_ts")).alias("HL"),
        (pl.col("k2_press_ts") - pl.col("k1_release_ts")).alias("IL"),
        (pl.col("k2_press_ts") - pl.col("k1_press_ts")).alias("PL"),
        (pl.col("k2_release_ts") - pl.col("k1_release_ts")).alias("RL"),
        pl.col("k1_press_ts").alias("key1_timestamp")
    ]).select(
        # Ensure selection matches FEATURE_SCHEMA order and names
        "user_id", "platform_id", "video_id", "session_id", "sequence_id",
        "key1", "key2", "HL", "IL", "PL", "RL", "key1_timestamp",
        "k1_press_idx_original", "k1_release_idx_original", 
        "k2_press_idx_original", "k2_release_idx_original"
    )
    
    # Cast to final schema to ensure consistency
    final_features_df = final_features_df.select(
        [pl.col(c).cast(FEATURE_SCHEMA[c]) for c in FEATURE_SCHEMA.keys()]
    )


    log_df = pl.DataFrame(log_breaks_list, schema_overrides=LOG_SCHEMA) if log_breaks_list else None
    if log_df is not None and log_df.is_empty(): log_df = None
    
    return final_features_df, log_df


def discover_files(raw_dir: str, verbose: bool = True) -> List[Tuple[int, str]]:
    """
    Discovers all valid user data files that meet the criteria.

    Args:
        raw_dir: The root directory containing user subdirectories.
        verbose: If True, print diagnostic messages.

    Returns:
        A list of tuples, where each tuple is (user_id, filepath).
    """
    user_args_list: List[Tuple[int, str]] = []
    bad_users: List[int] = []

    if not os.path.exists(raw_dir):
        print(f"Error: Raw data directory not found: {raw_dir}")
        return []

    user_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if verbose:
        print(f"Found {len(user_dirs)} potential user directories.")

    for user_id_str in sorted(user_dirs):
        try:
            user_id = int(user_id_str)
        except ValueError:
            if verbose:
                print(f"Skipping directory '{user_id_str}': not a valid integer user ID.")
            continue

        user_dir_path = os.path.join(raw_dir, user_id_str)
        actual_files = [f for f in os.listdir(user_dir_path) if os.path.isfile(os.path.join(user_dir_path, f)) and f.endswith(".csv")]
        
        # Check if user has the complete set of 18 files
        current_user_files_count = 0
        expected_filenames_for_user: List[str] = []
        for p_id in VALID_PLATFORMS:
            for s_id in VALID_SESSIONS:
                for v_id in VALID_VIDEOS:
                    expected_filename = f"{p_id}_{v_id}_{s_id}_{user_id}.csv" # Corrected: platform_video_session_user
                    expected_filenames_for_user.append(expected_filename)
                    if expected_filename in actual_files:
                        current_user_files_count +=1
        
        if current_user_files_count == EXPECTED_FILES_PER_USER:
            for filename in expected_filenames_for_user: # Iterate through expected to ensure they exist
                 if filename in actual_files: # Double check, though count match implies it
                    user_args_list.append((user_id, os.path.join(user_dir_path, filename)))
        else:
            bad_users.append(user_id)
            if verbose:
                print(f"User {user_id} has {current_user_files_count} files, expected {EXPECTED_FILES_PER_USER}. Skipping.")

    if verbose:
        print(f"Found {len(user_args_list) // EXPECTED_FILES_PER_USER if EXPECTED_FILES_PER_USER > 0 else 0} users with complete data.")
        if bad_users:
            print(f"Found {len(bad_users)} users with incomplete data: {bad_users}")
    
    return user_args_list

def main_processing_loop(raw_dir: str, save_dir: str, use_multiprocessing: bool = True, num_workers: Optional[int] = None, verbose: bool = True):
    """
    Orchestrates the discovery, processing, and saving of keystroke data.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        if verbose:
            print(f"Created save directory: {save_dir}")

    files_to_process = discover_files(raw_dir, verbose=verbose)
    if not files_to_process:
        print("No files found to process. Exiting.")
        return

    if num_workers is None:
        num_workers = os.cpu_count() # Default to all available cores

    all_features_list: List[pl.DataFrame] = []
    all_logs_list: List[pl.DataFrame] = []

    if use_multiprocessing and len(files_to_process) > 1 : # Multiprocessing only beneficial for multiple files
        if verbose:
            print(f"Starting processing of {len(files_to_process)} files with {num_workers} workers...")
        # Ensure num_workers does not exceed number of files for small batches
        actual_workers = min(num_workers, len(files_to_process)) if num_workers else len(files_to_process)
        
        with multiprocessing.Pool(processes=actual_workers) as pool:
            results = list(tqdm(pool.imap_unordered(process_single_file_polars, files_to_process), total=len(files_to_process), desc="Processing files"))
    else:
        if verbose:
            print(f"Starting sequential processing of {len(files_to_process)} files...")
        results = [process_single_file_polars(args) for args in tqdm(files_to_process, desc="Processing files")]

    for features_df, logs_df in results:
        if features_df is not None and not features_df.is_empty():
            all_features_list.append(features_df)
        if logs_df is not None and not logs_df.is_empty():
            all_logs_list.append(logs_df)

    # --- Save combined results ---
    if all_features_list:
        combined_features_df = pl.concat(all_features_list)
        features_save_path = os.path.join(save_dir, "typenet_features.csv")
        combined_features_df.write_csv(features_save_path)
        if verbose:
            print(f"Saved {len(combined_features_df)} features to {features_save_path}")
    else:
        if verbose:
            print("No features were extracted from any file.")
            # Create an empty features file with schema for consistency
            pl.DataFrame(schema=FEATURE_SCHEMA).write_csv(os.path.join(save_dir, "typenet_features.csv"))


    if all_logs_list:
        combined_logs_df = pl.concat(all_logs_list)
        logs_save_path = os.path.join(save_dir, "processing_logs.csv")
        combined_logs_df.write_csv(logs_save_path)
        if verbose:
            print(f"Saved {len(combined_logs_df)} log entries to {logs_save_path}")
    else:
        if verbose:
            print("No processing logs were generated.")
            # Create an empty logs file with schema
            pl.DataFrame(schema=LOG_SCHEMA).write_csv(os.path.join(save_dir, "processing_logs.csv"))


if __name__ == "__main__":
    # --- Multiprocessing setup (important for scripts, esp. on Windows/macOS) ---
    # This should be done only once at the start of the script execution.
    # 'spawn' is generally safer for cross-platform compatibility and avoiding issues
    # with shared resources or non-picklable objects in forked processes.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # This might happen if it's already set or if the context doesn't allow it (e.g. some nested scenarios)
        print("Note: Multiprocessing start method already set or context does not allow changing it.")
        pass

    # --- Configuration for the main run ---
    # Replace with your actual directory paths
    # Example: current_directory/data_dump/loadable_Combined_HU_HT
    base_dir = os.path.dirname(os.path.abspath(__file__)) # Or use os.getcwd() if script is run from its location
    
    # Construct raw_data_dir relative to the script's location or use an absolute path
    # Assuming data_dump is at the same level as the script's parent directory, or adjust as needed
    # For example, if script is in .../project/scripts/ and data is in .../project/data_dump/
    # raw_data_dir_relative = os.path.join(os.path.dirname(base_dir), "data_dump", "loadable_Combined_HU_HT")
    
    # For simplicity, assuming 'data_dump' is in the same directory as the script.
    # Modify this to your actual data directory structure.
    raw_data_dir = os.path.join(base_dir, "data_dump", "loadable_Combined_HU_HT")
    if not os.path.exists(raw_data_dir):
         # Fallback for common case where data_dump might be in parent of script's dir
        raw_data_dir = os.path.join(os.path.dirname(base_dir), "data_dump", "loadable_Combined_HU_HT")
        if not os.path.exists(raw_data_dir):
            # Fallback if data_dump is two levels up (e.g. script in src/processing/, data in data_dump/)
            raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(base_dir)), "data_dump", "loadable_Combined_HU_HT")
            if not os.path.exists(raw_data_dir):
                 print(f"Error: Raw data directory not found. Tried multiple common locations based on script path. Please check path: {raw_data_dir}")
                 exit(1) # Exit if data directory is crucial and not found


    hostname = socket.gethostname()
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S") # Removed Min and Sec suffix for brevity
    processed_data_save_dir = os.path.join(base_dir, f"processed_data-{now}-{hostname}")

    print(f"Raw data input directory: {raw_data_dir}")
    print(f"Processed data output directory: {processed_data_save_dir}")

    # Run the main processing loop
    main_processing_loop(
        raw_dir=raw_data_dir,
        save_dir=processed_data_save_dir,
        use_multiprocessing=True,  # Set to False to debug sequentially
        num_workers=None,          # None uses os.cpu_count()
        verbose=True
    )

    print("Processing complete.")