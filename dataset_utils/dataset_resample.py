import glob
import librosa
from classifier.globals import FREQUENCY
import os

DATASET_PATH_MUSIC = ''
DATASET_PATH_SPEECH = ''

RES_DATASET_PATH_MUSIC = ''
RES_DATASET_PATH_SPEECH = ''


def resample(audio_files, output_directory):
    counter, fin = 0, len(audio_files)
    for filepath in audio_files:
        if counter % 5 == 0:
            print(counter, '/', fin)
        counter += 1
        y, sr = librosa.load(filepath, sr=FREQUENCY)  # Downsample
        if y.shape[0] == 2:
            y = librosa.to_mono(y)
        split_char = '\\' if os.name == 'nt' else '/'
        new_filename = f'.'.join(filepath.split(split_char)[-1].split('.'))
        path = f'{output_directory}{split_char}{new_filename}'
        librosa.output.write_wav(path, y, sr)

def run():
    speech_files, music_files = glob.glob(f'{DATASET_PATH_SPEECH}/*.wav'), glob.glob(f'{DATASET_PATH_MUSIC}/*.wav')
    resample(speech_files, RES_DATASET_PATH_SPEECH)
    resample(music_files, RES_DATASET_PATH_MUSIC)

