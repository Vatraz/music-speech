from matplotlib import pyplot as plt
import glob
from functions import read_audio_file
from random import shuffle
from prettytable import PrettyTable
from globals import DATASET_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD


def draw_plots(features=None):
    if features is None:
        features = []
    for feature in features:
        plt.figure(num=f"{feature} - {music_data['filepath']} - {speech_data['filepath']}")
        wv_m = music_data[feature]
        wv_s = speech_data[feature]
        plt.plot(music_data['timestamps'], wv_m, label='Muzyka')
        plt.plot(speech_data['timestamps'], wv_s, label='Mowa')
        plt.legend(loc='upper left')

    plt.show()


def draw_feature_table(feature_list):
    table = PrettyTable(['Feature', 'Music', 'Speech'])
    for feature in feature_list:
        if feature in ['filepath']:
            table.add_row([feature, music_data[feature], speech_data[feature]])
        else:
            table.add_row([feature, format(music_data[feature], 'e'), format(speech_data[feature], 'e')])
    print(table)


if __name__ == '__main__':
    speech_files, music_files = glob.glob(DATASET_PATH + 'speech_25_16k/*.wav'), \
                                glob.glob(DATASET_PATH + 'music_25_16k/*.wav')
    shuffle(speech_files)
    shuffle(music_files)

    music_data = read_audio_file(music_files[0], FRAME_WIDTH, ZCR_THRESHOLD)
    speech_data = read_audio_file(speech_files[0], FRAME_WIDTH, ZCR_THRESHOLD)

    plt.plot(range(len(music_data['wavedata'])), music_data['wavedata'], range(len(music_data['wavedata'])), speech_data['wavedata'])
    draw_feature_table(['filepath', 'zcr_mean', 'zcr_diff_mean','zcr_exceed_th',
                        'zcr_third_central_moment', 'zcr_std_of_fod', 'ste_mean'])
    draw_plots(features=['zcr', 'ste'])
