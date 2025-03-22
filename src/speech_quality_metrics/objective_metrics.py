
import os
import argparse
from glob import glob
from os.path import join, basename, isdir
import torch, torchaudio
from torchaudio.functional import resample
from torch.nn.functional import pad
from tqdm import tqdm
from torchaudio.pipelines import SQUIM_OBJECTIVE

class ObjectiveMetricsPredictor:
    def __init__(self, device=None):        
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.objective_predictor = SQUIM_OBJECTIVE.get_model().to(self.device)
        self.objective_predictor.eval()

    def _load_file(self, filepath: str)->(torch.Tensor, int):
        try:
            waveform, sr = torchaudio.load(filepath)
            assert sr >= 16000, "Sample rate must be at least 16kHz"                
            if sr != 16000:
                waveform = resample(waveform, sr, 16000)
                sr = 16000
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
        except Exception as e:
            print(f"Error loading file {filepath}: {str(e)}")
            return False
        return waveform, sr
    
    def __call__(self, input_str)->float:
        if isdir(input_str):
            return self.predict_folder(input_str)
        else:
            return self.predict_file(input_str)

    def predict_file(self, filepath: str)->float:
        loaded_result = self._load_file(filepath)
        if not loaded_result:
            print(f"Failed to load file: {filepath}")
            return None
        waveform, sr = loaded_result
        with torch.no_grad():
            stoi, pesq, si_sdr = self.objective_predictor(waveform.to(self.device))
        return {
            "stoi": stoi[0].to('cpu').item(),
            "pesq": pesq[0].to('cpu').item(),
            "si_sdr": si_sdr[0].to('cpu').item()
        }          

    def _collate_fn(self, batch):
        # pad sequences to have same length
        batch = [item.squeeze() for item in batch if item is not None]
        max_length = max([item.shape[0] for item in batch])
        batch = [pad(item, (0, max_length - item.shape[0])) for item in batch]
        return torch.stack(batch, dim=0)
    
    def _predict_batch(self, fileslist):
        loaded_data = self._load_file(fileslist[0])
        if loaded_data is None:
            return None
        if not loaded_data:
            return None
        _, sr = loaded_data
        batch_list = [self._load_file(filepath)[0].squeeze() for filepath in fileslist]
        batch = self._collate_fn(batch_list)
        with torch.no_grad():
            stoi_hyp, pesq_hyp, si_sdr_hyp = self.objective_predictor(batch.to(self.device))
        return stoi_hyp.to('cpu'), pesq_hyp.to('cpu'), si_sdr_hyp.to('cpu')
    
    def _create_batchs(self, fileslist, batch_size=1):
        for i in range(0, len(fileslist), batch_size):
            yield fileslist[i:i+batch_size]

    def predict_folder(self, dirpath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        scores = {}
        filelist = sorted(glob(join(dirpath, search_str)))
        total_batches = (len(filelist) + batch_size - 1) // batch_size
        for batch in tqdm(self._create_batchs(filelist, batch_size), total=total_batches, desc="Processing Batches"):
            metrics = self._predict_batch(batch)
            if metrics is None:
                continue
            for i, filepath in enumerate(batch):
                stoi = metrics[0][i].item()
                pesq = metrics[1][i].item()
                si_sdr = metrics[2][i].item() if metrics[2][i].item() > 0 else 0
                #score = (stoi + (pesq / 4.5) + (si_sdr / 35.0) ) / 3.0
                scores[filepath] = {
                    "stoi": metrics[0][i].item(),
                    "pesq": metrics[1][i].item(),
                    "si_sdr": metrics[2][i].item()#,
                    #"score": score
                }                    
        return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, default='samples/0', help='Input folder')
    parser.add_argument('--output_file',    '-o', type=str, default='metrics_prediction.json', help='Output json filepath')
    parser.add_argument('--batch_size',     '-b', type=int, default=1, help='Batch size for processing files in parallel')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',         '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    quality_predictor = ObjectiveMetricsPredictor(args.device)    
    scores = quality_predictor.predict_folder(args.input_dir, args.batch_size, args.search_pattern)

    import json
    with open(args.output_file, "w") as ofile:
        json.dump(scores, ofile, indent=4)


if __name__ == "__main__":
    main()
