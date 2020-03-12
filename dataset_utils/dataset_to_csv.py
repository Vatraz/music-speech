import glob
import numpy as np
import pandas as pd

from classifier.globals import FRAME_WIDTH, WINDOW_WIDTH,  NUM_FRAMES_WINDOW, OUT_PATH
from classifier.functions import get_input_vector
from classifier.audio_segment import MyAudioSegment

DATASET_PATH_MUSIC = ''
DATASET_PATH_SPEECH = ''
OUT_NAME = ''


def get_audio_features(filepath, sound_type):
    segment = MyAudioSegment.from_wav(filepath)
    samples = segment.get_ndarray_of_samples()
    vector_list = []
    for index in range(0, samples.size + 1 - WINDOW_WIDTH, WINDOW_WIDTH):
        vector = get_input_vector(samples[index:index+WINDOW_WIDTH], FRAME_WIDTH, NUM_FRAMES_WINDOW)
        vector_list.append(np.concatenate(([sound_type], vector[0])))
    return vector_list


def run():
    speech_files, music_files = glob.glob(f'{DATASET_PATH_SPEECH}/*.wav'), glob.glob(f'{DATASET_PATH_MUSIC}/*.wav')

    music_data, speech_data = [], []

    for file in music_files:
        music_data.extend(get_audio_features(file, sound_type=1))

    for file in speech_files:
        speech_data.extend(get_audio_features(file, sound_type=0))

    for data, ext in [(music_data+speech_data, ''), (music_data, 'm'), (speech_data, 's')]:
        df = pd.DataFrame(data)
        df.columns = ['type', 'zcr_diff_mean', 'zcr_third_central_moment', 'zcr_exceed_th', 'zcr_std_of_fod', 'ste_mler']
        df.to_csv(f'{OUT_PATH}{OUT_NAME}_{ext}.csv', index=None, header=True)
