import os
import polars as pl


# def filter_IL_top_k(df: pl.DataFrame, level: str = 'platform_id', k: int=10) -> pl.DataFrame:
#     """
#     Filter the DataFrame to keep only the top k records for each platform based on IL.
#     """
#     if level == 'user_id':
#         label_features = ["user_id"]
#     elif level == 'platform_id':
#         label_features = ["user_id",  "platform_id"]
#     elif level == 'session_id':
#         label_features = ["user_id", "platform_id","session_id"]
#     elif level == "video_id":
#         label_features = ["user_id", "platform_id","session_id","video_id"]
#     else:
#         raise ValueError("level must be either 'platform_id' or 'user_id'")


#     # Get IL features
#     df_filtered  = df[ label_features + [col for col in df.columns if col.startswith("IL_")]]


#     # Get top-k features
#     df_top_k_features = df_filtered.group_by("key1", "key2").count().sort("count", descending=True).head(k)

#     # Get top k IL pairs
#     df_top_k = df_filtered.filter(
#         (pl.col("key1").is_in(df_top_k_features['key1'].to_list())) &
#         (pl.col("key2").is_in(df_top_k_features['key2'].to_list())) &
#         (pl.col("valid"))
#     )

#     return df_top_k


def filter_IL_top_k(
    df: pl.DataFrame, level: str = "platform_id", k: int = 10
) -> pl.DataFrame:
    """
    Filter the DataFrame to keep only the top k records for each platform based on IL.
    """
    if level == "user_id":
        label_features = ["user_id"]
    elif level == "platform_id":
        label_features = ["user_id", "platform_id"]
    elif level == "session_id":
        label_features = ["user_id", "platform_id", "session_id"]
    elif level == "video_id":
        label_features = ["user_id", "platform_id", "session_id", "video_id"]
    else:
        raise ValueError("level must be either 'platform_id' or 'user_id'")

    # Get IL features
    return df[label_features + [col for col in df.columns if col.startswith("IL_")]]


if __name__ == "__main__":

    for imp_path in ["imputation_global", "imputation_user"]:
        dpath = f"ml_experiments/{imp_path}"

        for file, level in zip(
            ["dataset_1_full.csv", "dataset_2_full.csv", "dataset_3_full.csv"],
            ["platform_id", "session_id", "video_id"],
        ):
            filepath = os.path.join(dpath, file)
            print(f"Loading {filepath}")

            # Load the dataset
            df = pl.read_csv(filepath, has_header=True)

            df_filtered = filter_IL_top_k(df, level=level, k=10)

            save_file_path = os.path.join(
                dpath, file.replace(".csv", "_IL_filtred.csv")
            )
            print(f"Saving filtered data as\n\t{save_file_path}")
            df_filtered.write_csv(save_file_path, include_header=True)
