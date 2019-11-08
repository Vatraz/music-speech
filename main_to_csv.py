import pandas as pd
import glob
from functions import get_audio_features
from globals import DATASET_PATH, OUT_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD, DS_NAME

if __name__ == '__main__':
    speech_files, music_files = glob.glob(DATASET_PATH + 'speech' + DS_NAME + '_25_16k/*.wav'), \
                                glob.glob(DATASET_PATH + 'music' + DS_NAME + '_25_16k/*.wav')

    music_data = [get_audio_features(file, FRAME_WIDTH, ZCR_THRESHOLD, sound_type='M') for file in music_files]
    speech_data = [get_audio_features(file, FRAME_WIDTH, ZCR_THRESHOLD, sound_type='S') for file in speech_files]
    print('zrobione?')
    df = pd.DataFrame(music_data + speech_data)
    # df = df[df.ste_mler != 0.0]
    df.to_csv(OUT_PATH + DS_NAME + '.csv', index=None, header=True)
