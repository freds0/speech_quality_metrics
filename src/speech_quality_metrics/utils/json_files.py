import os
import json
import pandas as pd
from typing import Dict, List, Optional

def load_json(json_file: str) -> pd.DataFrame:
    """
    Load a DataFrame from a JSON file.

    Parameters:
    json_file (str): Path to the JSON file.

    Returns:
    pd.DataFrame: Loaded DataFrame.
    """
    try:
        with open(json_file, "r") as jfile:
            data = json.load(jfile)
        df = pd.DataFrame(data).transpose().reset_index()
        return df
    except Exception as e:
        print(f"Error loading file {json_file}: {e}")
        raise

def get_partial_path(path: str) -> str:
    """
    Extract the last two components of the path.

    Parameters:
    path (str): Full path.

    Returns:
    str: Last two components of the path.
    """
    parts = path.split(os.sep)
    new_path = os.sep.join(parts[-2:])
    return new_path

def load_json_files(filelist: List[str], namelist: List[str]) -> List[pd.DataFrame]:

    # Load and process JSON files
    dataframes = []
    for ifile in filelist:
        print(f"Loading file {ifile}")
        try:
            df = load_json(ifile)
            dataframes.append(df)
        except Exception as e:
            print(f"Error processing file {ifile}: {e}")

    for index, df in enumerate(dataframes):
        df["index"] = df["index"].apply(lambda x: get_partial_path(x))
        df["model"] = namelist[index]
    
    df = pd.concat(dataframes, axis=0)
    return df
