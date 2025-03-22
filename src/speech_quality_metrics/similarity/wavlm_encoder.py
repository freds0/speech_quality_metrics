from os.path import join, basename
from tqdm import tqdm
from glob import glob
import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
from torch.nn.functional import pad, cosine_similarity
from .abstract_encoder import AbsEncoder

class WavlmEncoder(AbsEncoder):
    '''
    Source: https://huggingface.co/microsoft/wavlm-base-plus-sv
    '''        
    def __init__(self, device = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
        self.model = self.model.eval()
        self.model = self.model.to(self.device)

    def _create_batchs(self, fileslist, batch_size=1):
        for i in range(0, len(fileslist), batch_size):
            yield fileslist[i:i+batch_size]

    '''
    def _collate_fn(self, batch):
        # pad sequences to have same length
        batch = [item.squeeze() for item in batch if item is not None]
        max_length = max([item.shape[0] for item in batch])
        batch = [pad(item, (0, max_length - item.shape[0])) for item in batch]
        return torch.stack(batch, dim=0)
    '''

    def _encode(self, filepath):
        waveform, sr = self._load_file(filepath)
        inputs = self.feature_extractor(waveform.squeeze(), padding=True, return_tensors="pt", sampling_rate = 16000)
        inputs = inputs.to(self.device)
        with torch.no_grad():
            embeddings = self.model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
        return embeddings

    def _encode_batch(self, fileslist):
        batch_list = [self._load_file(filepath)[0].squeeze().numpy() for filepath in fileslist]        
        inputs_batch= self.feature_extractor(
            batch_list,
            padding=True,
            return_tensors="pt",
            sampling_rate = 16000
        ).to(self.device)
        with torch.no_grad():
            embeddings_batch = self.model(**inputs_batch).embeddings
            embeddings_batch = torch.nn.functional.normalize(embeddings_batch, dim=-1).cpu()
        return embeddings_batch

    def predict_similarity(self, filepath, filepath_gt):
        emb = self._encode(filepath)
        emb_gt = self._encode(filepath_gt)
        return cosine_similarity(emb.squeeze(), emb_gt.squeeze(), dim=-1).item()

    '''
    def predict_folder(self, dirpath: str, gt_filepath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        scores = {}
        filelist = sorted(glob(join(dirpath, search_str)))
        filelist.insert(0, gt_filepath)
        for batch in tqdm(self._create_batchs(filelist, batch_size)):
            embeddings = self._encode_batch(batch)
            for i, filepath in enumerate(batch):
                sim = cosine_similarity(embeddings[i], embeddings[0], dim=-1).item()
                #
                #threshold = 0.86  # the optimal threshold is dataset-dependent
                #if sim < threshold:
                #    print("Speakers are not the same!")                
                #
                scores[filepath] = {
                    "sim": sim
                }                    
        return scores        
    '''

    def predict_folder(self, dirpath: str, gt_dirpath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        filelist = sorted(glob(join(dirpath, search_str)))
        gt_filelist = sorted(glob(join(gt_dirpath, search_str)))
        assert len(filelist) == len(gt_filelist), f"Number of files in {dirpath} and {gt_dirpath} do not match"
        #wavs_list = [preprocess_wav(filepath) for filepath in filelist]
        #gt_wavs_list = [preprocess_wav(filepath) for filepath in gt_filelist]
        #embeddings = np.array([self.encoder.embed_utterance(wav) for wav in wavs_list])
        #gt_embeddings = np.array([self.encoder.embed_utterance(wav) for wav in gt_wavs_list])
        scores = {}
        #for i, filepath in tqdm(enumerate(filelist)):
        for batch, batch_gt in tqdm(zip(self._create_batchs(filelist, batch_size), self._create_batchs(gt_filelist, batch_size))):
            embeddings = self._encode_batch(batch)
            gt_embeddings = self._encode_batch(batch_gt)
            for i, filepath in enumerate(batch):                                    
                sim = cosine_similarity(embeddings[i], gt_embeddings[i], dim=-1).item()
                filename = basename(filepath)
                scores[filename] = {
                    "sim": sim#.tolist()
                } 
        return scores           
