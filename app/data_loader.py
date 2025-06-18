import os
import pandas as pd


def load_data(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data, or None if the file is not found.
    """
    # Check if the file exists before trying to load it
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        return None
    
    # Use a try-except block to handle potential errors during file reading
    try:
        df = pd.read_csv(file_path)
        print(f"\nSuccessfully loaded {os.path.basename(file_path)} with {len(df)} rows.")
        return df
    except Exception as e:
        print(f"Error loading file '{file_path}': {e}")
        return None

