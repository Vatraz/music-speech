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
    speech_files, music_files = (DATASET_PATH + 'music_wav_22k/jazz.wav',
                                 #  DATASET_PATH + 'speech_16k/shannon.wav')
                                 'jezu - Copy (2).wav')

    music_data = read_audio_file(music_files, FRAME_WIDTH, ZCR_THRESHOLD)
    speech_data = read_audio_file(speech_files, FRAME_WIDTH, ZCR_THRESHOLD)

    #  plt.plot(range(len(music_data['wavedata'])), music_data['wavedata'], range(len(music_data['wavedata'])), speech_data['wavedata'])
    #  draw_feature_table(['filepath', 'ste_mean', 'ste_diff_mean','ste_exceed_th',
    #                      'ste_third_central_moment', 'ste_std_of_fod', 'ste_mean'])
    #  draw_plots(features=['ste', 'ste'])

    #  # tylko dla dlugich STE
    #  music_data['timestamps'] = music_data['timestamps'][int(len(music_data['timestamps'])//1.2):]
    #  music_data['ste'] = music_data['ste'][int(len(music_data['timestamps'])//1.2):]
    

    lng = len(music_data['ste']) if len(music_data['ste']) <= len(speech_data['ste']) else len(speech_data['ste'])
    # lng = lng // 3
    x = list(range(lng))
    max = max(music_data['ste'])
    music_data['ste'] = [14 * d / max for d in  music_data['ste']]
    speech_data['ste'] = [14 * d / max for d in  speech_data['ste']]
    plt.subplot(2,1,1)
    plt.title('a) ste dla mowy')
    plt.plot(x, music_data['ste'][:lng], c='r')
    # plt.ylim(0,12)
    plt.ylabel('ste')
    plt.xlabel('Numer ramki')

    plt.subplot(2,1,2)
    plt.title('b) ste dla utworu muzycznego')
    plt.plot(x, speech_data['ste'][:lng], c='b')
    plt.ylim(0,12)
    plt.ylabel('ste')
    plt.xlabel('Numer ramki')

    plt.show()



    plt.subplot(1,2,1)
    plt.title('a) mowa')
    plt.hist(music_data['ste'][:lng], bins=20, alpha=0.7, color='r', edgecolor='r')
    plt.ylabel('liczność')
    plt.xlabel('wartość ste')

    plt.subplot(1,2,2)
    plt.title('b) utwór muzyczny')
    plt.hist(speech_data['ste'][:lng], bins=20, alpha=0.7, color='b', edgecolor='b')
    plt.ylabel('liczność')
    plt.xlabel('wartość ste')

    plt.show()
