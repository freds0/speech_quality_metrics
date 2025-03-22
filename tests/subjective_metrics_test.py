import os
from speech_quality_metrics.subjective_metrics import SubjectiveMetricsPredictor

predictor = SubjectiveMetricsPredictor(device='cpu')

input_dir = 'samples/0'

results = predictor.predict_folder(input_dir)

print(results)
