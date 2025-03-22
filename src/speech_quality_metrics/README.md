
# Speech Quality Metrics

This README provides detailed instructions for setting up and effectively utilizing the Speech Quality Metrics script. The `main.py` script allows the computation of multiple audio quality metrics including objective, subjective, speaker similarity, intelligibility, and silence metrics.

## Overview

The script utilizes multiple predictor classes from the `speech_quality_metrics` package to assess audio files. It supports selecting a specific metric to compute or running all metrics in combination.

## Requirements

Ensure your Python environment includes:
- Python 3.9 or higher
- Necessary libraries from `speech_quality_metrics` package

## Features

- Supports multiple audio quality assessments:
  - Objective Metrics
  - Subjective Metrics
  - Speaker Similarity
  - Intelligibility Metrics
  - Silence Metrics
- Can compute multiple metrics simultaneously.
- Outputs results in JSON format.

## Installation

To use the script, ensure all dependencies are installed, including the `speech_quality_metrics` package and any dependencies specific to the models used (e.g., PyTorch, pydub, torchmetrics).


```bash
pip3 install torch torchvision torchaudio

```

```bash
pip install -e .
```

## Usage

### Command Line Arguments


- `--input_dir`, `-i`: Directory containing the audio files to be processed.
- `--gt_audio_file`, `-a`: Ground truth audio file for speaker similarity.
- `--gt_text_file`, `-g`: Ground truth text file for intelligibility metrics
- `--batch_size`, `-b`: Batch size for inference.
- `--search_pattern`, `-s`: Search pattern for files in input folder (default='*.wav').
- `--output_file`, `-o`: File path where the output JSON will be saved.
- `--threshold`, `-t`: Threshold for silence metrics. Tweak based on signal-to-noise ratio (default=-80).
- `--interval`, `-n`: Interval in ms used in silence metrics. Increase to speed up (default=1).
- `--device`, `-d`: Device to run the model on: cpu | cuda.
- `--metric`, `-m`: Specify which metrics to calculate. Options include:
  - `obj` for Objective Metrics
  - `sub` for Subjective Metrics
  - `int` for Intelligibility Metrics
  - `spk` for Speaker Similarity Metrics
  - `sil` for Silence Metrics
  - `all` to run all available metrics

### Running the Script

To run the script for a specific metric:

```bash
python main.py --input_dir path/to/audio_files --output_file path/to/output.json --metric obj
```

To compute all metrics:

```bash
python main.py --input_dir path/to/audio_files --output_file path/to/output.json --metric all
```

### Example Output

The output JSON will detail the computed metrics for each audio file processed:

```json
{
    "sample1.wav": {
        "objective": {"MOS": 3.5, "PESQ": 2.1},
        "subjective": {"MOS": 4.0},
        "silence": {"duration": 120, "silence": 40, "percentage": 0.33}
    },
    "sample2.wav": {
        "intelligibility": {"WER": 0.15, "CER": 0.05},
        "speaker_similarity": {"score": 0.85}
    }
}
```

Each section of the output corresponds to the selected metrics for each audio file.

## Scripts Overview

### 1. `objective_metrics.py`
This script calculates various objective audio quality metrics such as PESQ, STOI, and SI-SDR. It supports both individual audio files and directories of audio files.

#### Features

- Single file and directory processing
- Automatic resampling to 16000 Hz
- Handling of stereo files by averaging channels
- Batch processing for efficiency
- Output results in JSON format


#### Command Line Arguments
- `--input_dir`, `-i`: Specify the directory containing audio files to be processed.
- `--output_file`, `-o`: Specify the file path where the output JSON will be saved.
- `--batch_size`, `-b`: Batch size for processing multiple files in parallel.
- `--search_pattern`, `-s`: File pattern to search for in the input directory (default: `*.wav`).
- `--device`, `-d`: Specify the computation device (`cpu` or `cuda`).


#### Usage

To run the script, use the following command:

```bash
python objective_metrics.py --input_dir path/to/audio_files --output_file path/to/output.json --batch_size 10 --search_pattern "*.wav" --device cuda
```

This command processes audio files in `path/to/audio_files`, saving the metrics in `path/to/output.json`, processing 10 files at a time on the CUDA device.


### Example Output

The output JSON will contain metrics for each processed file. Here is an example of what the output might look like:

```json
{
    "sample1.wav": {
        "stoi": 0.85,
        "pesq": 2.5,
        "si_sdr": 7.1
    },
    "sample2.wav": {
        "stoi": 0.78,
        "pesq": 1.8,
        "si_sdr": 5.4
    }
}
```

Each entry provides the STOI, PESQ, and SI-SDR values along with a combined score for the audio file.


### 2. `subjective_metrics.py`

This script calculates various Mean Opinion Score (MOS). It supports both individual audio files and directories of audio files. The `SubjectiveMetricsPredictor` class utilizes a pretrained model from the SpeechMOS project to predict the MOS for audio quality. 

## Features

- Resampling to 22050 Hz for non-compliant audio samples
- Mono channel processing by averaging multi-channel audio
- Batch processing for efficiency
- Output results in JSON format

## Usage

### Command Line Arguments

- `--input_dir`, `-i`: Directory containing the audio files.
- `--output_file`, `-o`: File path where the 'output' JSON will be saved.
- `--batch_size`, `-b`: Number of files to process in parallel.
- `--search_pattern`, '-s': Pattern to match files in the directory (default: `*.wav`).
- `--device`, `-d': Computation device (`cpu` or `cuda`).

### Running the Script

Use the following command to process files:

```bash
python subjective_metrics.py --input_dir path/to/audio_files --output_file path/to/output.json --batch_size 5 --search_pattern "*.wav" --device cuda
```

### Example Output

The output will be a JSON file containing the MOS predictions:

```json
{
    "sample1.wav": {"mos": 3.5},
    "sample2.wav": {"mos": 4.2}
}
```

Each entry lists the MOS value for the corresponding audio file.

### 3. `similarity_metrics.py`

This script evaluates speaker similarity across audio files using several libraries. It extracts voice embeddings and calculates their similarities, providing insights into how similar the voices in different audio files are. The `SpeakerSimilarityPredictor` class facilitates the evaluation of speaker similarity, offering a choice between three different models: `ResemblyzerEncoder`, `EcapaTdnnVox2Encoder`, `WavlmEncoder`, and `Ecapa2Encoder`. It can handle both individual file comparisons and batch processing of directories.

#### Features

- Supports multiple speaker encoder models.
- Compatible with CUDA for GPU acceleration.
- Batch processing for large datasets.
- Outputs results in JSON format.

#### Command Line Arguments

- `--input_dir`, `-i`: Specify the directory containing noisy audio samples.
- `--gt`, `-g`: Path to the ground truth audio file or directory.
- `--model`, `-m`: Choose from `resemblyzer`, `ecapa_tdnn`, `wavlm`, or `ecapa2`.
- `--output_file`, `-o`: Specify the path for the output JSON file.
- `--batch_size`, `-b`: Number of files to process simultaneously.
- `--search_pattern`, `-s`: File pattern to match in the directory.
- `--device`, `-d`: Computation device (`cpu` or `cuda`).

#### Usage

To run the script, use the following command pattern:

```bash
python speaker_similarity.py --input_dir path/to/noisy_samples --gt path/to/ground_truth.wav --model resemblyzer --output_file path/to/results.json --batch_size 5 --search_pattern "*.wav" --device cuda
```

### 4. `intelligibility_metrics.py`
This script calculates the Word Error Rate (WER) and Character Error Rate (CER) for automatic speech recognition (ASR) transcriptions using pre-trained models. The `IntelligibilityMetricsPredictor` class utilizes ASR models to transcribe audio files and compute intelligibility metrics against ground truth transcriptions. It supports processing audio files from a specified directory.

#### Features

- Uses pre-trained ASR models for transcription.
- Calculates WER and CER between automated and ground truth transcriptions.
- Handles directory inputs for batch processing.
- Outputs results in JSON format.

#### Usage

##### Command Line Arguments

- `--input_dir`, `-i`: Directory containing the audio files.
- `--gt_text_file`, `-g`: Optional path to a file containing ground truth transcriptions.
- `--output_file`, `-o`: File path where the output JSON will be saved.
- `--device`, `-d`: Computation device (`cpu` or `cuda`).

##### With Grount Truth 
If you have the ground truth text, run the script using the command:

```bash
python intelligibility_metrics.py --input_dir path/to/audio --gt_text_file path/to/ground_truth --output_file path/to/output.json
```
##### Without Grount Truth 

If you don't have the ground truth text, the script will transcribe the audio and calculate the WER and CER. Run the script using the command:

```bash
python intelligibility_metrics.py --input_dir path/to/audio --output_file path/to/output.csv
```


#### Example Output

The output will be a JSON file containing WER and CER metrics:

```json
{
    "sample1.wav": {
        "wer": 0.23,
        "cer": 0.05
    },
    "sample2.wav": {
        "wer": 0.29,
        "cer": 0.07
    }
}
```

Each entry provides the WER and CER for the corresponding audio file against its ground truth transcription.


### 5. `silence_metrics.py`
This script  measures the duration of silence within audio files based on a specified decibel threshold. The `SilenceMetricsPredictor` class analyzes audio files to determine the proportion of silence. It can process both single files and entire directories, making it suitable for batch analysis.

#### Features

- Calculates the total duration of silence and the percentage of silence in each audio file.
- Configurable silence threshold and interval duration for analysis granularity.
- Supports batch processing of multiple audio files.

#### Usage

##### Command Line Arguments

- `--input_dir`, `-i`: Directory containing the audio files.
- `--output_file`, `-o`: File path where the output JSON will be saved.
- `--search_pattern`, '-s': File pattern to match files in the directory (default: `*.wav`).
- `--threshold`, '-t': Decibel threshold for silence detection. Adjust based on your specific signal-to-noise ratio.
- `--interval`, '-n': Interval in milliseconds for analyzing audio chunks. Smaller intervals may provide more precise measurements but require more processing time.

### Running the Script

To run the script, use the following command:

```bash
python silence_metrics.py --input_dir path/to/audio_files --output_file path/to/output.json --threshold -80 --interval 1
```

### Example Output

The output will be a JSON file detailing the silence metrics for each processed file:

```json
{
    "sample1.wav": {
        "duration": 10.0,
        "silence": 3.0,
        "perc": 30.0
    },
    "sample2.wav": {
        "duration": 8.0,
        "silence": 2.4,
        "perc": 30.0
    }
}
```

Each entry lists the total duration of the audio, the silence duration, and the percentage of silence.


### 6.  ploting_graphs.py
This script processes JSON files to generate distribution plots and statistical reports for different models. It calculates descriptive statistics and confidence intervals and provides a summary report for each model.

#### Features

- Calculates and prints descriptive statistics and confidence intervals for each model.
- Generates distribution plots based on the data in JSON files.


#### Usage

##### Command Line Arguments

- `files`: List of JSON files to be processed. These files should contain data in JSON format.
- `--names`: List of names to associate with each JSON file. The number of names must match the number of files.
- `--output_dir`: Directory where the distribution plots will be saved. If not provided, plots will not be saved.
### Running the Script

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


### 7.  `statistics_report.py`
This script processes JSON files to generate statistical reports for different models. It calculates descriptive statistics and confidence intervals and prints a summary report for each model.

#### Features

- Calculates and prints descriptive statistics and confidence intervals for each specified model.
- Processes multiple JSON files, associating each file with a specific model name.

#### Usage

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