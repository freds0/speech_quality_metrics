import os
from speech_quality_metrics.similarity_metrics import SpeakerSimilarityPredictor

input_dir = 'samples/0'
gt_filepath = 'samples/1/1.wav'
#models = ['resemblyzer', 'ecapa_tdnn', 'ecapa2', 'wavlm']
models = ['resemblyzer', 'wavlm']

for model in models:
    predictor = SpeakerSimilarityPredictor(model=model, device='cpu')
    results = predictor.predict_folder(input_dir, gt_filepath, batch_size=1, search_str='*.wav')
    print(results)
