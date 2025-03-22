
# Quality Metrics

This README provides detailed instructions for setting up and effectively utilizing speech quality metrics.

## Project Description

`speech_quality_metrics` is a package that offers various speech quality metrics, including MOS (Mean Opinion Score) and PESQ (Perceptual Evaluation of Speech Quality).

## Installation

### Requirements

Before installing the package, ensure you have the following requirements:

- Python 3.9 or higher
- `pip` installed

### Package Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/freds0/speech_quality_metrics
   cd speech_quality_metrics
   ```

2. Install the dependencies:

   ```bash
   pip install torch torchvision torchaudio
   pip install -e .
   ```

## Usage

To use the speech_quality_metrics library, follow the instructions below to set up and execute various types of speech quality predictions:

### Objective Metrics

For evaluating objective metrics of speech quality:

```bash
from speech_quality_metrics.objective_metrics import ObjectiveMetricsPredictor
obj_quality_predictor = ObjectiveMetricsPredictor(device)    
objective_metrics = obj_quality_predictor.predict_folder(input_dir, batch_size, search_pattern)
```

#### Parameters
- `input_dir`: Specify the directory containing audio files to be processed.
- `batch_size`: Batch size for processing multiple files in parallel.
- `search_pattern`: File pattern to search for in the input directory (default: `*.wav`).
- `device`: Specify the computation device (`cpu` or `cuda`).


### Subjective Metrics

To assess subjective metrics of speech quality:

```bash
from speech_quality_metrics.subjective_metrics import SubjectiveMetricsPredictor
sbj_quality_predictor = SubjectiveMetricsPredictor(device)
subjective_metrics = sbj_quality_predictor.predict_folder(input_dir, batch_size, search_pattern)
```

#### Parameters
- `input_dir`: Specify the directory containing audio files to be processed.
- `batch_size`: Batch size for processing multiple files in parallel.
- `search_pattern`: File pattern to search for in the input directory (default: `*.wav`).
- `device`: Specify the computation device (`cpu` or `cuda`).



### Intelligibility Metrics

For measuring speech intelligibility:

```bash
from speech_quality_metrics.intelligibility_metrics import IntelligibilityMetricsPredictor
int_quality_predictor = IntelligibilityMetricsPredictor(device)
intelligibility_metrics = int_quality_predictor.predict_folder(input_dir, gt_text_file, search_pattern)
```

#### Parameters
- `input_dir`: Specify the directory containing audio files to be processed.
- `gt_text_file`: Optional path to a file containing ground truth transcriptions.
- `search_pattern`: File pattern to search for in the input directory (default: `*.wav`).
- `device`: Specify the computation device (`cpu` or `cuda`).


### Speaker Similarity Metrics

To determine speaker similarity metrics:

```bash
from speech_quality_metrics.similarity_metrics import SpeakerSimilarityPredictor
spk_quality_predictor = SpeakerSimilarityPredictor(model, device)
similarity_metrics = spk_quality_predictor.predict_folder(input_dir, gt_dir, batch_size, search_pattern)
```

#### Parameters
- `input_dir`: Specify the directory containing audio samples.
- `gt_dir`: Path to the ground truth audio file or directory.
- `model`: Choose from `resemblyzer`, `ecapa_tdnn`, `wavlm` or `ecapa2`.
- `batch_size`: Number of files to process simultaneously.
- `search_pattern`: File pattern to match in the directory.
- `device`: Computation device (`cpu` or `cuda`).


### Silence Metrics

For analyzing silence within speech files:

```bash
from speech_quality_metrics.silence_metrics import SilenceMetricsPredictor
sil_quality_predictor = SilenceMetricsPredictor(threshold, interval)
silence_metrics = sil_quality_predictor.predict_folder(input_dir, search_pattern)
```

#### Parameters

- `threshold`: Decibel threshold for silence detection. Adjust based on your specific signal-to-noise ratio.
- `interval`: Interval in milliseconds for analyzing audio chunks. Smaller intervals may provide more precise measurements but require more processing time.
- `input_dir`: Directory containing the audio files.
- `search_pattern`: File pattern to match files in the directory (default: `*.wav`).


These examples demonstrate how to use different components of the speech_quality_metrics library to evaluate various aspects of speech quality across a collection of audio files.
