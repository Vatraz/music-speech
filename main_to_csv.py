import glob
import pandas as pd
from functions import get_audio_features
from globals import DATASET_PATH, OUT_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD, DS_NAME


def main(DS_NAME):
    print('-> "', DS_NAME, '"')
    if DS_NAME == '':
        speech_files, music_files = glob.glob(DATASET_PATH + 'speech_25_16k/*.wav'), \
                                    glob.glob(DATASET_PATH + 'music_25_16k/*.wav')
    else:
        speech_files, music_files = glob.glob(DATASET_PATH + 'speech_' + DS_NAME + '_25_16k/*.wav'), \
                                    glob.glob(DATASET_PATH + 'music_' + DS_NAME + '_25_16k/*.wav')

    music_data = [get_audio_features(file, FRAME_WIDTH, sound_type=1) for file in music_files]
    speech_data = [get_audio_features(file, FRAME_WIDTH, sound_type=0) for file in speech_files]
    print('zrobione?')
    df = pd.DataFrame(music_data + speech_data)
    # df = df[df.ste_mler != 0.0]
    if DS_NAME == '':
        DS_NAME = 'zero'

    df.to_csv(OUT_PATH + DS_NAME + '.csv', index=None, header=True)
    df = pd.DataFrame(music_data)
    df.to_csv(OUT_PATH + DS_NAME + 'm.csv', index=None, header=True)
    df = pd.DataFrame(speech_data)
    df.to_csv(OUT_PATH + DS_NAME + 's.csv', index=None, header=True)


DS_NAME = 't'
main(DS_NAME)
DS_NAME = 'g'
main(DS_NAME)
DS_NAME = 'r'
main(DS_NAME)
DS_NAME = ''
main(DS_NAME)