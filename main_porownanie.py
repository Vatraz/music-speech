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
        plt.plot(music_data['timestamps'], wv_m, label='music')
        plt.plot(speech_data['timestamps'], wv_s, label='speech')
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
    #  speech_files, music_files = (DATASET_PATH + 'music_16k/redhot.wav',
    speech_files, music_files = (DATASET_PATH + 'music_y_16k/hopeless.wav',
                                 #  DATASET_PATH + 'speech_16k/shannon.wav')
                                 DATASET_PATH + 'speech_16k/male.wav')

    music_data = read_audio_file(music_files, FRAME_WIDTH, ZCR_THRESHOLD)
    speech_data = read_audio_file(speech_files, FRAME_WIDTH, ZCR_THRESHOLD)

    #  plt.plot(range(len(music_data['wavedata'])), music_data['wavedata'], range(len(music_data['wavedata'])), speech_data['wavedata'])
    #  draw_feature_table(['filepath', 'zcr_mean', 'zcr_diff_mean','zcr_exceed_th',
    #                      'zcr_third_central_moment', 'zcr_std_of_fod', 'ste_mean'])
    #  draw_plots(features=['zcr', 'ste'])

    #  # tylko dla dlugich STE
    #  music_data['timestamps'] = music_data['timestamps'][int(len(music_data['timestamps'])//1.2):]
    #  music_data['ste'] = music_data['ste'][int(len(music_data['timestamps'])//1.2):]
    

    lng = len(music_data['timestamps']) if len(music_data['timestamps']) <= len(speech_data['timestamps']) else len(speech_data['timestamps'])
    lng = lng//3
    x = list(range(lng))

    plt.subplot(2,1,1)
    plt.title('a) STE dla mowy')
    plt.plot(x, music_data['ste'][:lng])
    plt.ylim(0,15)
    plt.ylabel('STE')
    plt.xlabel('Numer badanej ramki')
    
    plt.subplot(2,1,2)
    plt.title('b) STE dla utowru muzycznego')
    plt.plot(x, speech_data['ste'][:lng])
    plt.ylim(0,15)
    plt.ylabel('STE')
    plt.xlabel('Numer badanej ramki')

    plt.show()



    #  plt.subplot(1,2,1)
    #  plt.title('a) mowa')
    #  plt.hist(music_data['ste'][:lng*3], bins=30)
    #
    #  plt.subplot(1,2,2)
    #  plt.title('b) utwór muzyczny')
    #  plt.hist(speech_data['ste'][:lng*3], bins=30)
    #
    #  plt.show()
