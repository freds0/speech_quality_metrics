from abc import ABC, abstractmethod
import torch
import torchaudio

class AbsEncoder(ABC):
    @abstractmethod
    def predict_similarity(self, filepath, filepath_gt):
        pass
    
    @abstractmethod
    def predict_folder(self, dirpath: str, gt_filepath: str, batch_size : int = 1, file_ext : str = 'wav') -> dict:
        pass   
    
    def _convert_to_mono(self, waveform):
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def _convert_sr(self, waveform, sr):
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        return waveform, 16000    
    
    def _load_file(self, filepath):        
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] == 2:
            waveform = self.convert_to_mono(waveform)
        if sr != 16000:
            waveform, sr = self._convert_sr(waveform, sr)
        return waveform, sr        