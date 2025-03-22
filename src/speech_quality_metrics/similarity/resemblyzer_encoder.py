from .abstract_encoder import AbsEncoder
from resemblyzer import preprocess_wav, VoiceEncoder
import torch
from torch.nn.functional import cosine_similarity, pad
import numpy as np
from os.path import join, basename
from glob import glob
from tqdm import tqdm

class ResemblyzerEncoder(AbsEncoder):
    '''
    Source: https://github.com/resemble-ai/Resemblyzer/blob/master/demo01_similarity.py
    '''        
    def __init__(self, device = None):
        self.encoder = VoiceEncoder()

    def predict_similarity(self, filepath, filepath_gt):
        wav = preprocess_wav(filepath)     
        wav_gt = preprocess_wav(filepath_gt)       
        emb = self.encoder.embed_utterance(wav)
        emb_gt = self.encoder.embed_utterance(wav_gt)
        return np.inner(emb, emb_gt)

    '''        
    def predict_folder(self, dirpath: str, gt_filepath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        filelist = sorted(glob(join(dirpath, search_str)))
        filelist.insert(0, gt_filepath)
        wavs_list = [preprocess_wav(filepath) for filepath in filelist]
        embeddings = np.array([self.encoder.embed_utterance(wav) for wav in wavs_list])
        scores = {}
        for i, filepath in enumerate(filelist):
            sim = np.inner(embeddings[i], embeddings[0])
            scores[filepath] = {
                "sim": sim.tolist()
            } 
        return scores
    '''        
    def predict_folder(self, dirpath: str, gt_dirpath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        filelist = sorted(glob(join(dirpath, search_str)))
        gt_filelist = sorted(glob(join(gt_dirpath, search_str)))
        assert len(filelist) == len(gt_filelist), f"Number of files in {dirpath} and {gt_dirpath} do not match"
        wavs_list = [preprocess_wav(filepath) for filepath in filelist]
        gt_wavs_list = [preprocess_wav(filepath) for filepath in gt_filelist]
        embeddings = np.array([self.encoder.embed_utterance(wav) for wav in wavs_list])
        gt_embeddings = np.array([self.encoder.embed_utterance(wav) for wav in gt_wavs_list])
        scores = {}
        for i, filepath in tqdm(enumerate(filelist)):
            sim = np.inner(embeddings[i], gt_embeddings[i])
            filename = basename(filepath)
            scores[filename] = {
                "sim": sim.tolist()
            } 
        return scores    
