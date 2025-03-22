from speech_quality_metrics.silence_metrics import SilenceMetricsPredictor

predictor = SilenceMetricsPredictor()

input_dir = 'samples/0'

results = predictor.predict_folder(input_dir)

print(results)
