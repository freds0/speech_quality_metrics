import argparse
from tqdm import tqdm
from glob import glob
from os.path import basename, join
from speechbrain.pretrained import EncoderASR
import os
import torch

class SpeechBrainTranscriptor:
    '''
    SpeechBrain Transcriptor for ASR using a pre-trained model with GPU support.
    Source: https://huggingface.co/speechbrain
    '''
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()

    def _load_model(self):
        """
        Load the SpeechBrain pre-trained ASR model and move it to the desired device.
        """
        model = EncoderASR.from_hparams(
            source="speechbrain/asr-wav2vec2-commonvoice-14-ar",
            savedir="pretrained_models/asr-wav2vec2-commonvoice-14-ar",
            run_opts={"device": self.device}
        )
        # Move model to the specified device
        model.device = self.device
        return model

    def transcribe(self, input_filepath):
        """
        Transcribe a single audio file.

        Parameters:
        - input_filepath: Path to the input audio file.
        
        Returns:
        - A string containing the transcription.
        """
        try:
            # Transcribe the audio file (automatically uses the specified device)
            transcription = self.model.transcribe_file(input_filepath)
            return transcription.strip()
        except Exception as e:
            # Print an error message if transcription fails
            print(f"Error processing {input_filepath}: {e}")
            return None

    def transcribe_folder(self, input_dir, output_file, search_pattern='*.wav'):
        """
        Transcribe all audio files in a folder and save the results to a CSV file.

        Parameters:
        - input_dir: Directory containing audio files.
        - output_file: Path to the output CSV file.
        - search_pattern: Pattern to search for audio files (default: '*.wav').
        """
        with open(output_file, 'w', encoding='utf-8') as ofile:
            for input_filepath in tqdm(sorted(glob(join(input_dir, search_pattern)))):
                transcription = self.transcribe(input_filepath)
                if transcription:
                    line = "{}|{}".format(basename(input_filepath), transcription)
                    ofile.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, default='wavs', help='Input folder')
    parser.add_argument('--output_file',    '-o', type=str, default='transcription.csv', help='Output CSV file')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',         '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    asr_model = SpeechBrainTranscriptor(device=args.device)
    asr_model.transcribe_folder(args.input_dir, args.output_file, args.search_pattern)


if __name__ == "__main__":
    main()
