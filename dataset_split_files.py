import numpy as np
import glob
from scipy.io import wavfile
from dataset_utils import check_directories
from globals import  DATASET_PATH

speech_files, music_files = glob.glob(DATASET_PATH + "speech_16k/*.wav"), \
                            glob.glob(DATASET_PATH + "music_16k/*.wav")


def split(audio_files, output_directory, out_samplerate):
    cnt, fin = 0, len(audio_files)
    for filepath in audio_files:
        if cnt % 5 == 0:
            print(cnt, '/',fin)
        cnt += 1
        file_samplerate, wavedata = wavfile.read(filepath)
        frame_width = int(file_samplerate * window_len)
        counter, start = 0, 0
        while start + frame_width <= len(wavedata):
            new_wavedata = wavedata[start:start+frame_width]
            new_filename = f'{counter}.'.join(filepath.split('/')[-1].split('.'))
            wavfile.write(f'{output_directory}/{new_filename}', file_samplerate, np.asarray(new_wavedata))
            # new_filename = f'{counter}.'.join(filepath.split('\\')[-1].split('.'))
            # wavfile.write(f'{output_directory}\\{new_filename}', file_samplerate, np.asarray(new_wavedata))
            counter += 1
            start += frame_width


out_samplerate = 22000  # Hz
window_len = 2.5

music_dir, speech_dir = check_directories('{}_test_25_22k')
split(music_files, music_dir, out_samplerate)
split(speech_files, speech_dir, out_samplerate)
