import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from typing import List
from speech_quality_metrics.utils.json_files import load_json_files
from matplotlib.ticker import MultipleLocator  # para controlar localização dos ticks

def distribution_plot(df: pd.DataFrame, title: str, x_axis_name: str, output_path: str) -> None:
    """
    Distribution plot using violin plots with horizontal dotted lines every 5 points on the y-axis.
    """

    # Get a list of unique models from the DataFrame
    unique_models = df['model'].unique().tolist()

    # Create a color palette with as many colors as there are models
    palette = sns.color_palette("Set2", n_colors=len(unique_models))

    # Dictionary that maps model name to a specific color
    color_map = {model_name: palette[i] for i, model_name in enumerate(unique_models)}

    # Create the figure
    plt.figure(figsize=(10, 6))

    # ---- VIOLIN PLOT ----
    sns.violinplot(
        data=df,
        x='model',
        y=x_axis_name,
        order=unique_models,
        palette=palette
    )

    # Title and axes labels
    plt.title(f'Distribution of {title}')
    plt.xlabel('Model')
    plt.ylabel(x_axis_name)

    # --- SETTING UP THE HORIZONTAL DOTTED LINES ---
    # Get current axes
    ax = plt.gca()

    # Define that major ticks on y-axis should appear every 5 units
    #ax.yaxis.set_major_locator(MultipleLocator(1))

    # Enable the grid on the y-axis, with a dotted line style
    ax.grid(axis='y', which='major', linestyle=':')

    # Adjust legend if needed
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

    # Load data from JSON files
    df = load_json_files(args.files, args.names)
    df = df.reset_index(drop=True)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Select columns to plot
    if args.columns:
        columns_to_plot = [col for col in args.columns if col in df.columns]
        if not columns_to_plot:
            print("No valid columns specified for plotting. Exiting.")
            return
    else:
        # Default: all columns except 'index' and 'model'
        columns_to_plot = [col for col in df.columns if col not in ['index', 'model']]

    # Generate a violin plot for each selected column
    for column in columns_to_plot:
        title = column.upper()
        output_path = os.path.join(args.output_dir, f"distribution_{title}.jpg")
        distribution_plot(df, title, column, output_path)

if __name__ == "__main__":
    main()
