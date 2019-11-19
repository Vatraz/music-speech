import numpy as np
import glob
from scipy.io import wavfile
from dataset_utils import check_directories
from globals import  DATASET_PATH, DS_NAME, FREQUENCY, WINDOW_LEN
from dataset_resample import resample, check_directories
from dataset_split_files import split


out_samplerate = FREQUENCY  # Hz
window_len = WINDOW_LEN
for DS_NAME in ('c'):
    speech_files, music_files = glob.glob(DATASET_PATH + 'speech_' + DS_NAME + "/*.wav"), \
                                glob.glob(DATASET_PATH + 'music_' + DS_NAME + "/*.wav")
    music_dir, speech_dir = check_directories('{}_' + DS_NAME + '_22k')
    resample(speech_files, speech_dir, out_samplerate)
    resample(music_files, music_dir, out_samplerate)

    speech_files, music_files = glob.glob(DATASET_PATH + 'speech_' + DS_NAME + '_22k/*.wav'), \
                                glob.glob(DATASET_PATH + 'music_' + DS_NAME + '_22k/*.wav')
    music_dir, speech_dir = check_directories('{}_' + DS_NAME + '_25_22k')
    split(music_files, music_dir, window_len)
    split(speech_files, speech_dir, window_len)