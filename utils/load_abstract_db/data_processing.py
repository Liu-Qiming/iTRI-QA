from pybtex.database import parse_string
import pandas as pd

def merge_dataframes(*dataframes, how='left', on='key'):
    """
    Merge multiple DataFrames on a specified column.

    Parameters:
        *dataframes (pd.DataFrame): DataFrames to merge.
        how (str): Type of merge to be performed. Default is 'left'.
        on (str): Column name to merge on. Default is 'key'.

    Returns:
        pd.DataFrame: Merged DataFrame.
    """
    if not dataframes:
        raise ValueError("At least one DataFrame must be provided for merging.")

    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on=on, how=how)

    return merged_df
