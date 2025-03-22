import csv
import argparse
from tqdm import tqdm
from os import makedirs
import torch
from os.path import join, exists, dirname, isdir, basename, splitext
import tempfile
from jiwer import wer, cer
from speech_quality_metrics.text.cleaners import basic_cleaners


class IntelligibilityMetricsPredictor:
    def __init__(self, device=None):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tmp_dir = tempfile.TemporaryDirectory().name
        #self.tmp_dir = "temp_results"
        self.tmp_gt_filepath = join(self.tmp_dir, "gt_transcriptions.csv")
        self.tmp_txt_filepath = join(self.tmp_dir, "txt_transcriptions.csv")

    def _transcribe_folder(self, input_dir, output_file=None, search_pattern='*.wav', model='wav2vec'):
        from speech_quality_metrics.asr.mms_transcriptor import MmsTranscriptor
        from speech_quality_metrics.asr.wav2vec_transcriptor import Wav2Vec2Transcriptor
        from speech_quality_metrics.asr.whisper_transcriptor import WhisperTranscriptor
        from speech_quality_metrics.asr.speechbrain_transcriptor import SpeechBrainTranscriptor

        output_file = output_file if output_file is not None else self.tmp_txt_filepath
        if not exists(dirname(output_file)):
            makedirs(dirname(output_file))

        if model == 'wav2vec':
            asr_model = Wav2Vec2Transcriptor(lang='en', device=self.device)
        elif model == 'mms':
            asr_model = MmsTranscriptor(model_name='fl102', lang="en", device=self.device)  
        elif model == 'speechbrain':
            asr_model = SpeechBrainTranscriptor(device=self.device)  
        else:
            asr_model = WhisperTranscriptor(device=self.device)

        asr_model.transcribe_folder(input_dir, output_file, search_pattern)
        return output_file

    def _transcribe_gt_folder(self, input_dir, output_file=None, search_pattern='*.wav'):
        from speech_quality_metrics.asr.whisper_transcriptor import WhisperTranscriptor
        output_file = output_file if output_file is not None else self.tmp_gt_filepath
        if not exists(dirname(output_file)):
            makedirs(dirname(output_file))
        asr_model = WhisperTranscriptor(device=self.device)
        asr_model.transcribe_folder(input_dir, output_file, search_pattern)
        return output_file

    def _get_wer(self, transcription, gt_text):
        #return self.wer(transcription.strip(), gt_text.strip()).item()
        try:
            if len(gt_text) == 0 and len(transcription) == 0:
                return 0.0
            elif len(gt_text) == 0 or len(transcription) == 0:
                return 1.0                
            return wer(transcription, gt_text)
        except Exception as e:
            print(f"Error calculatiing WER: {e}")
            return 1.0

    def _get_cer(self, transcription, gt_text):
        #return self.cer(transcription.strip(), gt_text.strip()).item()
        try:
            if len(gt_text) == 0 and len(transcription) == 0:
                return 0.0
            elif len(gt_text) == 0 or len(transcription) == 0:
                return 1.0        
            return cer(transcription, gt_text)
        except Exception as e:
            print(f"Error calculatiing CER: {e}")
            return 1.0

    def __call__(self, input_str)->float:
        if isdir(input_str):
            return self.predict_folder(input_str)
        else:
            return self.predict_file(input_str)
        return False
        
    def predict_from_files(self, txt_filepath, gt_filepath, model='wav2vec') -> dict:
        transcriptions_dict = self.load_transcriptions(txt_filepath)
        gt_transcriptions_dict = self.load_transcriptions(gt_filepath)
        scores = {}
        for filepath, text in tqdm(transcriptions_dict.items()):
            parent_directory = basename(dirname(filepath))
            file_id = join(parent_directory, basename(filepath))
            filename = splitext(basename(filepath))[0]
            if filename in gt_transcriptions_dict:
                text = basic_cleaners(text)
                gt_transcriptions_dict[filename] = basic_cleaners(gt_transcriptions_dict[filename])
                wer = self._get_wer(text, gt_transcriptions_dict[filename])
                cer = self._get_cer(text, gt_transcriptions_dict[filename])
                scores[file_id] = {
                    "gt": gt_transcriptions_dict[filename],
                    model: text, 
                    "wer": wer,
                    "cer": cer
                }                
            else:
                print(gt_transcriptions_dict)
                print(filename)
                print(f"Ground truth transcription not found for {filepath}")
        return scores

    def predict_folder(self, input_dir, gt_filepath: str = None, search_pattern: str = '*.wav', model: str = 'wav2vec') -> dict:
        if gt_filepath is None:
            gt_filepath = self._transcribe_gt_folder(input_dir, search_pattern=search_pattern)
        txt_filepath = self._transcribe_folder(input_dir, search_pattern=search_pattern, model=model)
        return self.predict_from_files(txt_filepath, gt_filepath, model=model)

    @staticmethod
    def load_transcriptions(text_file):
        # Load ground truth transcriptions
        transcriptions_dict = dict()
        with open(text_file, "r") as ifile:
            content_data = ifile.readlines()
        for line in content_data:
            filepath, text = line.strip().split("|")
            parent_directory = basename(dirname(filepath))
            #file_id = join(parent_directory, basename(filepath))
            file_id = splitext(basename(filepath))[0]
            transcriptions_dict[file_id] = text
        return transcriptions_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',    '-i', type=str, default='samples/0', help='Input folder')
    parser.add_argument('--gt_text_file', '-g', type=str, default=None, help='Ground truth text file')
    parser.add_argument('--output_file',  '-o', type=str, default='intelligibility_metrics.json', help='Output json filepath')
    parser.add_argument('--model',        '-m', type=str, default='wav2vec', help='Model transcriptor: wav2vec | mms | whisper')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',       '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    import json
    wer_predictor = IntelligibilityMetricsPredictor(args.device)
    wer_dict = wer_predictor.predict_folder(args.input_dir, args.gt_text_file, args.search_pattern, args.model)

    # Save the JSON file with characters correctly encoded
    with open(args.output_file, "w", encoding="utf-8") as ofile:
        json.dump(wer_dict, ofile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
