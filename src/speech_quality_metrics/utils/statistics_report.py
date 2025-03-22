import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import json
import os
from typing import Dict, List, Optional
from speech_quality_metrics.utils.json_files import load_json_files, get_partial_path

def confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Calculate the confidence interval for the provided data.

    Parameters:
    data (np.ndarray): Data to calculate the interval for.
    confidence (float): Confidence level.

    Returns:
    tuple: Confidence interval (lower, upper).
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin_of_error, mean + margin_of_error

def get_statistics_report(df: pd.DataFrame, model_name: str) -> None:
    """
    Calculate and print confidence intervals for a specific model.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    model_name (str): Name of the model for which to calculate the intervals.
    """
    df_filtered = df[df['model'] == model_name]
    describe = df_filtered.describe()

    ci: Dict[str, tuple] = {}
    for column in df_filtered.select_dtypes(include=[np.number]).columns:
        ci[column] = confidence_interval(df_filtered[column].dropna())

    # Print descriptive statistics and confidence intervals
    report =  f"{model_name.upper()} Statistics:\n"
    describe_str = describe.to_string()  
    report += describe_str
    report += f"\nConfidence intervals:\n {ci}\n"
    return report

def main() -> None:
    """
    Main function to process files and generate distribution plots.
    """
    parser = argparse.ArgumentParser(description="Process JSON files and generate distribution plots.")
    parser.add_argument(
        'files', 
        nargs='+',
        help='List of JSON files to be processed'
    )
    parser.add_argument(
        '--names',
        '-n',
        nargs='+',
        type=str,
        required=True,
        help='List of names to associate with the files'
    )    
    parser.add_argument(
        '--output_file',
        '-o',
        type=str,
        default='statistics_report.txt',
        help='Output filepath for the report'
    )
    args = parser.parse_args()

    assert len(args.files) == len(args.names), "The number of names must be equal to the number of files"

    df = load_json_files(args.files, args.names)
        
    for model in args.names:
        report = get_statistics_report(df, model)
        with open(args.output_file, 'a') as ofile:
            ofile.write(report)


if __name__ == "__main__":
    main()
