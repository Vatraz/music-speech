import numpy as np
import glob
from scipy.io import wavfile

speech_files, music_files = glob.glob("speech_wav/*.wav"), glob.glob("music_wav/*.wav")

sample_rate = 22050  # Hz
frame_len = 2.5
frame_width = int(sample_rate*2.5)


def split(audio_files, output_directory):
    audio_data = []

    for filepath in audio_files:
        samplerate, wavedata = wavfile.read(filepath)
        if sample_rate != samplerate:
            import sys
            sys.exit("Err")

        start = 0
        counter = 0
        while start + frame_width <= len(wavedata):
            new_wavedata = wavedata[start:start+frame_width]
            new_filename = f'{counter}.'.join(filepath.split('/')[-1].split('.'))
            wavfile.write(f'{output_directory}/{new_filename}', samplerate, np.asarray(new_wavedata))
            counter += 1
            start += frame_width

    return audio_data


music_data = split(music_files, 'music_24')
speech_data = split(speech_files, 'speech_24')
