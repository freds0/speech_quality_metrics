
import os
import argparse
from glob import glob
from os.path import join, isdir
import torch
import torchaudio
from torchaudio.functional import resample
from torch.nn.functional import pad
from tqdm import tqdm

class SubjectiveMetricsPredictor:
    '''
    https://github.com/tarepan/SpeechMOS
    '''
    def __init__(self, device=None):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mos_predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        self.mos_predictor.to(self.device)
        self.mos_predictor.eval()

    def _load_file(self, filepath: str)->(torch.Tensor, int):
        try:
            waveform, sr = torchaudio.load(filepath)
            #assert sr >= 22050, "Sample rate must be at least 22kHz"            
            if sr != 22050:
                waveform = resample(waveform, sr, 22050)
                sr = 22050
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)                
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return False
        return waveform, sr
    
    def __call__(self, input_str)->float:
        if isdir(input_str):
            return self.predict_folder(input_str)
        else:
            return self.predict_file(input_str)

    def predict_file(self, filepath: str)->float:
        waveform, sr = self._load_file(filepath)
        mos = self.mos_predictor(waveform.to(self.device), sr)
        return {
            "mos": mos.item()
        }  

    def _collate_fn(self, batch):
        # pad sequences to have same length
        batch = [item.squeeze() for item in batch if item is not None]
        max_length = max([item.shape[0] for item in batch])
        batch = [pad(item, (0, max_length - item.shape[0])) for item in batch]
        return torch.stack(batch, dim=0)
    
    def _predict_batch(self, fileslist):
        sr= self._load_file(fileslist[0])[1]
        batch_list = [self._load_file(filepath)[0] for filepath in fileslist]
        batch = self._collate_fn(batch_list)
        mos = self.mos_predictor(batch.to(self.device), sr)
        return mos

    def _create_batchs(self, fileslist, batch_size=1):
        for i in range(0, len(fileslist), batch_size):
            yield fileslist[i:i+batch_size]

    def predict_folder(self, dirpath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        scores = {}
        filelist = sorted(glob(join(dirpath, search_str)))
        if len(filelist) == 0:
            print(f"No files found in {dirpath} with search string {search_str}")
            return None
        total_batches = (len(filelist) + batch_size - 1) // batch_size
        for batch in tqdm(self._create_batchs(filelist, batch_size), total=total_batches, desc="Processing Batches"):        
            mos_score = self._predict_batch(batch)
            for i, filepath in enumerate(batch):
                scores[filepath] = {
                    "mos": mos_score[i].item()
                }
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, default='samples/0', help='Input folder')
    parser.add_argument('--output_file',    '-o', type=str, default='mos_prediction.json', help='Output json filepath')
    parser.add_argument('--batch_size',     '-b', type=int, default=1, help='Batch size for processing files in parallel')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',         '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    quality_predictor = SubjectiveMetricsPredictor(args.device)
    mos_dict = quality_predictor.predict_folder(args.input_dir, args.batch_size, args.search_pattern)

    import json
    with open(args.output_file, "w") as ofile:
        json.dump(mos_dict, ofile, indent=4)


if __name__ == "__main__":
    main()
