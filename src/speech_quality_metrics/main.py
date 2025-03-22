import argparse
import json
from speech_quality_metrics.objective_metrics import ObjectiveMetricsPredictor
from speech_quality_metrics.subjective_metrics import SubjectiveMetricsPredictor
from speech_quality_metrics.similarity_metrics import SpeakerSimilarityPredictor
from speech_quality_metrics.intelligibility_metrics import IntelligibilityMetricsPredictor
from speech_quality_metrics.silence_metrics import SilenceMetricsPredictor

def combine_dicts(dicts):
    combined_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in combined_dict:
                combined_dict[key] = {}
            combined_dict[key].update(value)
    return combined_dict

def obj_metrics(args):
    obj_quality_predictor = ObjectiveMetricsPredictor(args.device)
    result_metrics = obj_quality_predictor.predict_folder(args.input_dir, args.batch_size, args.search_pattern)
    return result_metrics

def int_metrics(args):
    int_quality_predictor = IntelligibilityMetricsPredictor(args.device)
    result_metrics = int_quality_predictor.predict_folder(args.input_dir, args.gt_text_file, args.search_pattern) 
    return result_metrics

def sub_metrics(args):
    sbj_quality_predictor = SubjectiveMetricsPredictor(args.device)
    result_metrics = sbj_quality_predictor.predict_folder(args.input_dir, args.batch_size, args.search_pattern)
    return result_metrics

def spk_metrics(args):
    if args.gt_audio_file is None:
        print("Ground truth audio file is required for speaker similarity metric. Especify with -a option.")
        return False    
    spk_quality_predictor = SpeakerSimilarityPredictor(args.spk_model, args.device)
    result_metrics = spk_quality_predictor.predict_folder(args.input_dir, args.gt_audio_file, args.batch_size, args.search_pattern)
    return result_metrics

def sil_metrics(args):
    sil_quality_predictor = SilenceMetricsPredictor(args.threshold, args.interval)    
    result_metrics = sil_quality_predictor.predict_folder(args.input_dir, args.search_pattern)
    return result_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, required=True)
    parser.add_argument('--gt_audio_file',  '-a', type=str, default=None, help='Ground truth audio file for speaker similarity')
    parser.add_argument('--gt_text_file',   '-g', type=str, default=None, help='Ground truth text file for intelligibility metrics')
    parser.add_argument('--output_file',    '-o', type=str, required=True)
    parser.add_argument('--batch_size',     '-b', type=int, default=1, help='Batch size for processing files in parallel')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')
    parser.add_argument('--spk_model',      '-e', type=str, default='resemblyzer', help='Speaker similarity model: resemblyzer | ecapa_tdnn | ecapa2')
    parser.add_argument('--threshold',      '-t', type=int, default=-80, help='Threshold for silence. Tweak based on signal-to-noise ratio')
    parser.add_argument('--interval',       '-n', type=int, default=1, help='Interval in ms. Increase to speed up')    
    parser.add_argument('--device',         '-d', type=str, default=None, help='Device to run the model on: cpu | cuda')    
    parser.add_argument('--metric',         '-m', type=str, default='ojb', help='Objective (obj) | subjective (sub) | intelligibility (int) | speaker similarity (spk) | silence (sil) metrics')
    args = parser.parse_args()

    result_metrics = False
    if args.metric == 'ojb':        
        result_metrics = obj_metrics(args)

    elif args.metric == 'int':
        result_metrics = int_metrics(args)

    elif args.metric == 'sub':
        result_metrics = sub_metrics(args)
    
    elif args.metric == 'spk':
        result_metrics = spk_metrics(args)

    elif args.metric == 'sil':
        result_metrics = sil_metrics(args)

    elif args.metric == 'all':
        obj_results = obj_metrics(args)
        int_resutls = int_metrics(args)
        sub_results = sub_metrics(args)
        spk_results = spk_metrics(args)
        sil_resutls = sil_metrics(args)
        result_metrics = combine_dicts([obj_results, int_resutls, sub_results, spk_results, sil_resutls])
    
    else:
        print(f"Metric {args.metric} not found. Available metrics: obj | sub | int | spk | sil | all")
        return False

    if result_metrics:   
        with open(args.output_file, "w") as ofile:
            json.dump(result_metrics, ofile, indent=4)


if __name__ == "__main__":
    main()
