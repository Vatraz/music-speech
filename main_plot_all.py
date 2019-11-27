import numpy as np
from matplotlib import pyplot as plt
import glob
from random import shuffle
from prettytable import PrettyTable
from functions import read_audio_file
from globals import DATASET_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD, DS_NAME


def analyse(audio_files):
    return [read_audio_file(filepath, FRAME_WIDTH, ZCR_THRESHOLD) for filepath in audio_files]


def draw_plots(features=None):
    for feature in features:
        plt.figure(num=feature)
        wv_m = [i[feature] for i in music_data]
        wv_s = [i[feature] for i in speech_data]
        plt.plot(list(range(num_files)), wv_m, label='music')
        plt.plot(list(range(num_files)), wv_s, label='speech')
        plt.legend(loc='upper left')

    plt.show()


def draw_mean_feature_table(features_list):
    table = PrettyTable(['', 'Music', 'Speech'])
    for ft in features_list:
        table.add_row([
            ft,
            format(np.mean([i[ft] for i in music_data]), 'e'),
            format(np.mean([i[ft] for i in speech_data]), 'e'),
        ])
    print(table)


if __name__ == '__main__':
    speech_files, music_files = glob.glob(DATASET_PATH + "speech_fm_25_16k/*.wav"), \
                                glob.glob(DATASET_PATH + "music_fm_25_16k/*.wav")

    shuffle(speech_files)
    shuffle(music_files)
    num_files = len(music_files)

    music_data = analyse(music_files)
    speech_data = analyse(speech_files)

    draw_mean_feature_table([
        'zcr_mean',
        'zcr_diff_mean',
        'zcr_exceed_th',
        'zcr_third_central_moment',
        'zcr_std_of_fod',
        'ste_mean',
        'ste_mler',
    ])

    draw_plots(features=['zcr_mean', 'ste_mean'])
