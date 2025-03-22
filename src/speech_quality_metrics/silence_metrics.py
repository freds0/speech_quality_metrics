import argparse
from pydub import AudioSegment
from glob import glob
from tqdm import tqdm
from os.path import join, isdir

class SilenceMetricsPredictor:

    def __init__(self, threshold=-80, interval=1):
        self.threshold = threshold
        self.interval = interval
    
    def _load_audio(self, filepath):
        "load audio file"
        return AudioSegment.from_wav(filepath)

    def __call__(self, input_str)->float:
        if isdir(input_str):
            return self.predict_folder(input_str)
        else:
            return self.predict_file(input_str)
        return False
                
    def get_silence(self, wav):       
        # break into chunks
        chunks = [wav[i : i + self.interval] for i in range(0, len(wav), self.interval)]
        # find number of chunks with dBFS below threshold
        silent_blocks = 0
        for c in chunks:
            if c.dBFS == float('-inf') or c.dBFS < self.threshold:
                silent_blocks += 1
        # convert blocks into seconds
        return round(silent_blocks * (self.interval / 1000), 3)

    def get_duration(self, wav):
        "get duration of audio in seconds"
        return round(len(wav) / 1000, 3)
    
    def predict_file(self, filepath: str):
        wav = self._load_audio(filepath) 
        duration = self.get_duration(wav)
        silence_dur = self.get_silence(wav)
        return {
            "duration": duration,
            "silence": silence_dur,
            "perc": round(silence_dur / duration, 3)
        }   
    
    def predict_folder(self, dirpath: str, search_pattern: str = "*.wav"):
        silence_values = {}
        for filepath in tqdm(glob(join(dirpath, search_pattern))):
            wav = self._load_audio(filepath) 
            duration = self.get_duration(wav)
            silence_dur = self.get_silence(wav)
            silence_values[filepath] = {                
                "duration": duration,
                "silence": silence_dur,
                "perc": round(silence_dur / duration, 3)
            }  
        return silence_values
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',      '-i', type=str, default='samples/0', help='Input folder')
    parser.add_argument('--output_file',    '-o', type=str, default='silence_prediction.json', help='Output json filepath')
    parser.add_argument('--search_pattern', '-s', type=str, default='*.wav', help='Search pattern for files in input folder')    
    parser.add_argument('--threshold',      '-t', type=int, default=-80, help='Threshold for silence. Tweak based on signal-to-noise ratio')
    parser.add_argument('--interval',       '-n', type=int, default=1, help='Interval in ms. Increase to speed up')
    args = parser.parse_args()

    sp = SilenceMetricsPredictor(args.threshold, args.interval)    
    silence_values = sp.predict_folder(args.input_dir, args.search_pattern)

    import json
    with open(args.output_file, "w") as ofile:
        json.dump(silence_values, ofile, indent=4)

    
if __name__ == "__main__":
    main()


