from speech_quality_metrics.objective_metrics import ObjectiveMetricsPredictor

predictor = ObjectiveMetricsPredictor(device='cpu')

input_dir = 'samples/0'

results = predictor.predict_folder(input_dir)

print(results)
