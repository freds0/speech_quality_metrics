# Ploting Graphs and Statistics Report

## ploting_graphs.py
This script processes JSON files to generate distribution plots and statistical reports for different models. It calculates descriptive statistics and confidence intervals and provides a summary report for each model.

### Features

- Calculates and prints descriptive statistics and confidence intervals for each model.
- Generates distribution plots based on the data in JSON files.


### Usage

#### Command Line Arguments

- `files`: List of JSON files to be processed. These files should contain data in JSON format.
- `--names`: List of names to associate with each JSON file. The number of names must match the number of files.
- `--output_dir`: Directory where the distribution plots will be saved. If not provided, plots will not be saved.
## Running the Script

To run the script, use the following command:

```bash
python ploting_graphs.py file1.json file2.json --names model1 model2 --output_dir /path/to/output
```

In this example:

- file1.json and file2.json are the JSON files to be processed.
- model1 and model2 are the names associated with each file.
- Distribution plots will be saved in /path/to/output.

### Example Output

The script generates a distribution plot for each numeric column in the provided JSON files.


## `statistics_report.py`
This script processes JSON files to generate statistical reports for different models. It calculates descriptive statistics and confidence intervals and prints a summary report for each model.

### Features

- Calculates and prints descriptive statistics and confidence intervals for each specified model.
- Processes multiple JSON files, associating each file with a specific model name.

### Usage

##### Command Line Arguments

- `files`: List of JSON files to be processed. These files should contain data in JSON format.
- `--names`: List of names to associate with each JSON file. The number of names must match the number of files.

To run the script, use the following command:

```bash
python statistics_report.py file1.json file2.json --names model1 model2 --output_dir output/filepath
```

In this example:

- file1.json and file2.json are the JSON files to be processed.
- model1 and model2 are the names associated with each file.
- Distribution plots will be saved in output/filepath.

### Example Output

The script generates a statistics report for each numeric column in the provided JSON files.
