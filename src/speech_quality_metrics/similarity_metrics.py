import argparse
from glob import glob
from os.path import join, isdir
import torch
#from .similarity.ecapa_tdnn_vox2_encoder import EcapaTdnnVox2Encoder
#from .similarity.ecapa2_encoder import Ecapa2Encoder
#from .similarity.resemblyzer_encoder import ResemblyzerEncoder
#from .similarity.wavlm_encoder import WavlmEncoder
from speech_quality_metrics.similarity.resemblyzer_encoder import ResemblyzerEncoder  # Absolute import
from speech_quality_metrics.similarity.wavlm_encoder import WavlmEncoder
from speech_quality_metrics.similarity.ecapa_tdnn_vox2_encoder import EcapaTdnnVox2Encoder

model_dict = {
    'resemblyzer': ResemblyzerEncoder,
    'ecapa_tdnn': EcapaTdnnVox2Encoder,
    #'ecapa2': Ecapa2Encoder,
    'wavlm': WavlmEncoder
}
class SpeakerSimilarityPredictor:

    def __init__(self, model='wavlm', device=None):
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model in model_dict:
            self.speaker_encoder = model_dict[model](self.device)
        else:
            print(f"Model {model} not found")
            return False  

    def __call__(self, input_str_gen, input_str_gt)->float:
        if isdir(input_str_gen) and isdir(input_str_gt):
            return self.predict_folder(input_str_gen, input_str_gt)
        else:
            return self.predict_similarity(input_str_gen, input_str_gt)
        return False
    
    def predict_similarity(self, filepath, filepath_gt):    
        return self.speaker_encoder.predict_similarity(filepath, filepath_gt)
    
    def predict_folder(self, dirpath: str, gt_filepath: str, batch_size : int = 1, search_str : str = '*.wav') -> dict:
        return self.speaker_encoder.predict_folder(dirpath, gt_filepath, batch_size, search_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, default='samples/0/', help='Input folder')
    parser.add_argument('--gt',             '-g', type=str, default='samples/gt/19_198_000000_000000.wav', help='Ground truth file')
    parser.add_argument('--model',          '-m', type=str, default='resemblyzer', help="Available Models: wavlm | resemblyzer | ecapa_tdnn | ecapa2")
    parser.add_argument('--output_file',    '-o', type=str, default='similarity_prediction.json', help='Output json filepath')
    parser.add_argument('--batch_size',     '-b', type=int, default=1, help='Batch size for processing files in parallel')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--device',         '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')
    args = parser.parse_args()

    sim_predictor = SpeakerSimilarityPredictor(args.model, args.device)
    if not sim_predictor:
        print(f"Model {args.model} not found")
        return False
    sim_dict =  sim_predictor.predict_folder(args.input_dir, args.gt, args.batch_size, args.search_pattern)

    import json
    with open(args.output_file, "w") as ofile:
        json.dump(sim_dict, ofile, indent=4)


if __name__ == "__main__":
    main()
