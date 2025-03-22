import torch
from huggingface_hub import hf_hub_download
import torchaudio
from .abstract_encoder import AbsEncoder

model_file = hf_hub_download(repo_id='Jenthe/ECAPA2', filename='ecapa2.pt', cache_dir=None)

class Ecapa2Encoder(AbsEncoder):
    '''
    Source: https://huggingface.co/Jenthe/ECAPA2
    '''        
    def __init__(self, device = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = torch.jit.load(model_file, map_location=self.device)
        #if self.device == "cuda":            
        #    self.model.half() # optional, but results in faster inference

    def encode(self, filepath)->float:
        waveform, sr = self._load_file(filepath)
        return self.model(waveform.unsqueeze(0).to(self.device))
    
    def encode_batch(self, batch)->float:
        return self.model(batch.to(self.device)) 
    
    