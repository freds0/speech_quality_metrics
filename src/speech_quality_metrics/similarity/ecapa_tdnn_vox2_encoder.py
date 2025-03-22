from tqdm import tqdm
from os.path import join, basename
from glob import glob
import torch
import torchaudio
from speechbrain.inference.interfaces import Pretrained
from speechbrain.inference.classifiers import EncoderClassifier
from torch.nn.functional import cosine_similarity
from torch.nn.functional import pad
from .abstract_encoder import AbsEncoder

class Encoder(Pretrained):
    '''
    Source: https://huggingface.co/yangwang825/ecapa-tdnn-vox2
    '''
    MODULES_NEEDED = [
        "compute_features",
        "mean_var_norm",
        "embedding_model"
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def encode_batch(self, wavs, wav_lens=None, normalize=False):
        # Manage single waveforms in input
        if len(wavs.shape) == 1:
            wavs = wavs.unsqueeze(0)

        # Assign full length if wav_lens is not assigned
        if wav_lens is None:
            wav_lens = torch.ones(wavs.shape[0], device=self.device)

        # Storing waveform in the specified device
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        wavs = wavs.float()

        # Computing features and embeddings
        feats = self.mods.compute_features(wavs)
        feats = self.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.mods.embedding_model(feats, wav_lens)
        if normalize:
            embeddings = self.hparams.mean_var_norm_emb(
                embeddings,
                torch.ones(embeddings.shape[0], device=self.device)
            )
        return embeddings


class EcapaTdnnVox2Encoder(AbsEncoder):
    def __init__(self, device=None):
        self.sr = 16000 # The system is trained with recordings sampled at 16kHz (single channel)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.speaker_encoder = Encoder.from_hparams(
                                   source="yangwang825/ecapa-tdnn-vox2"
                               )
    def _encode(self, filepath)->float:
        waveform, sr = self._load_file(filepath)
        enb = self.speaker_encoder.encode_batch(waveform.squeeze().to(self.speaker_encoder.device)) 
        return enb

    def predict_similarity(self, filepath, filepath_gt):
        embed = self._encode(filepath)        
        embed_gt = self._encode(filepath_gt)        
        return cosine_similarity(embed.squeeze(), embed_gt.squeeze(), dim=0).item()

    def _create_batchs(self, fileslist, batch_size=1):
        for i in range(0, len(fileslist), batch_size):
            yield fileslist[i:i+batch_size]

    def _collate_fn(self, batch):
        # pad sequences to have same length
        batch = [item.squeeze() for item in batch if item is not None]
        max_length = max([item.shape[0] for item in batch])
        batch = [pad(item, (0, max_length - item.shape[0])) for item in batch]
        return torch.stack(batch, dim=0)
    
    def _encode_batch(self, fileslist):
        sr= self._load_file(fileslist[0])[1]
        batch_list = [self._load_file(filepath)[0] for filepath in fileslist]
        batch = self._collate_fn(batch_list)
        return self.speaker_encoder.encode_batch(batch.to(self.device)) 
    '''
    def predict_folder(self, dirpath: str, gt_filepath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        scores = {}
        filelist = sorted(glob(join(dirpath, search_str)))
        filelist.insert(0, gt_filepath)
        for batch in tqdm(self._create_batchs(filelist, batch_size)):
            embeddings = self._encode_batch(batch)
            for i, filepath in enumerate(batch):
                sim = cosine_similarity(embeddings[i], embeddings[0]).item()
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
