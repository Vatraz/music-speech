import pandas as pd
import glob
from random import shuffle
from functions import get_audio_features
from globals import DATASET_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD
from timeit import default_timer as timer
if __name__ == '__main__':
    speech_files, music_files = glob.glob(DATASET_PATH + 'speech_musan_25_16k/*.wav'), \
                                glob.glob(DATASET_PATH + 'music_musan_25_16k/*.wav')

    for file in music_files:
        start = timer()
        get_audio_features(file, FRAME_WIDTH, ZCR_THRESHOLD, sound_type='M')
        end = timer()
        print(end - start)
        break
    music_data = [get_audio_features(file, FRAME_WIDTH, ZCR_THRESHOLD, sound_type='M') for file in music_files]
    speech_data = [get_audio_features(file, FRAME_WIDTH, ZCR_THRESHOLD, sound_type='S') for file in speech_files]

    df = pd.DataFrame(music_data + speech_data)
    df.to_csv('dataset_musan.csv', index=None, header=True)
