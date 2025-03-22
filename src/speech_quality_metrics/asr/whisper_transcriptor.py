import argparse
from tqdm import tqdm
from glob import glob
from os.path import basename, join
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch
import torchaudio
import torchaudio.transforms as T

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class WhisperTranscriptor:
    '''
    Source: https://huggingface.co/openai/whisper-large-v3
    '''
    def __init__(self, device=None):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.processor, self.model = self._load_model()

    def _load_model(self):
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        processor = AutoProcessor.from_pretrained(model_id)
        return processor, model.to(self.device)

    def _speech_file_to_array_fn(self, filepath, target_sample_rate=16000):
        waveform, sample_rate = torchaudio.load(filepath)
        if sample_rate != target_sample_rate:
            resampler = T.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        return waveform.squeeze()

    def transcribe(self, input_filepath):
        waveform = self._speech_file_to_array_fn(input_filepath)
        input_features = self.processor(waveform, sampling_rate=16000, return_tensors="pt").input_features
        # Convert input_features to the same dtype as the model
        input_features = input_features.to(self.device).to(torch_dtype)

        with torch.no_grad():
            predicted_ids = self.model.generate(input_features.to(self.device))

        text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return text[0]

    def transcribe_folder(self, input_dir, output_file, search_pattern='*.wav'):
        ofile = open(output_file, 'w')
        for input_filepath in tqdm(sorted(glob(join(input_dir, search_pattern)))):
            transcription = self.transcribe(input_filepath)
            line = "{}|{}".format(basename(input_filepath), transcription.strip())
            ofile.write(line + "\n")
        ofile.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, default='samples/noisy', help='Input folder')
    parser.add_argument('--output_file',    '-o', type=str, default='transcription_whisper.csv', help='Output csv file')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',         '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    asr_model = WhisperTranscriptor(args.device)
    asr_model.transcribe_folder(args.input_dir, args.output_file, args.search_pattern)


if __name__ == "__main__":
    main()

