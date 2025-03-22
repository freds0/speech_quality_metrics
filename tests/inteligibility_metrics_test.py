from speech_quality_metrics.intelligibility_metrics import IntelligibilityMetricsPredictor

predictor = IntelligibilityMetricsPredictor(device='cpu')

input_dir = 'samples/0'

results = predictor.predict_folder(input_dir)

print(results)
