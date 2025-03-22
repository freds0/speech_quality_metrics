import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import List
from speech_quality_metrics.utils.json_files import load_json_files

'''
def distribution_plot(df: pd.DataFrame, title: str, x_axis_name: str, output_path: str) -> None:
    """
    Plot and save the distribution of specified variables.

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    title (str): Title of the plot.
    x_axis_name (str): Name of the column to be plotted on the x-axis.
    output_path (str): Path where the plot will be saved.
    """
    if df.index.duplicated().any():
        print("Warning: Duplicated indices found. Resetting index.")
        df = df.reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=x_axis_name, hue="model", fill=True)

    plt.title(f'Distribution of {title}')
    plt.xlabel(x_axis_name)
    plt.ylabel('Density')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
'''
def distribution_plot(df: pd.DataFrame, title: str, x_axis_name: str, output_path: str) -> None:
    """
    Simplified distribution plot:
    - KDE plot separated by 'model'
    - Only a vertical line for the mean of each model
    - Prints mean and standard deviation for each model in the console
    """
    # Create the figure
    plt.figure(figsize=(10, 6))
    
    # Plot the distribution (KDE) for each model
    sns.kdeplot(data=df, x=x_axis_name, hue="model", fill=True)
    
    # Plot a vertical line for the mean of each model
    for i, (model_name, group) in enumerate(df.groupby("model")):
        mean_val = group[x_axis_name].mean()
        std_val = group[x_axis_name].std()
        
        # Print the mean and std to the console
        print(f"Model: {model_name}, Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        
        color = f'C{i}'  # Distinct color for each model
        plt.axvline(
            mean_val,
            color=color,
            linestyle='--',
            label=f'{model_name} Mean: {mean_val:.2f}'
        )

    # Title and labels
    plt.title(f'Distribution of {title}')
    plt.xlabel(x_axis_name)
    plt.ylabel('Density')
    plt.legend(loc='best')
    
    # Save and close
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


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
        '--output_dir',
        '-o',
        type=str,
        default='output',
        help='Output directory for the plots'
    )
    parser.add_argument(
        '--columns',
        '-c',
        nargs='*',
        type=str,
        default=None,
        help='List of columns to plot (if not specified, plots all columns except index and model)'
    )
    args = parser.parse_args()

    assert len(args.files) == len(args.names), "The number of names must be equal to the number of files"

    df = load_json_files(args.files, args.names)
    df = df.reset_index(drop=True) 

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Select columns to plot
    if args.columns:
        columns_to_plot = [col for col in args.columns if col in df.columns]
        if not columns_to_plot:
            print("No valid columns specified for plotting. Exiting.")
            return
    else:
        # Default to all columns except 'index' and 'model'
        columns_to_plot = [col for col in df.columns if col not in ['index', 'model']]

    for column in columns_to_plot:
        title = column.upper()
        output_path = os.path.join(args.output_dir, f"distribution_{title}.jpg")
        distribution_plot(df, title, column, output_path)


if __name__ == "__main__":
    main()
