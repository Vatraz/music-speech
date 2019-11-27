from pydub import AudioSegment
from functions import *
from scipy.io import wavfile
import sys
import time
import numpy as np

song = AudioSegment.from_file('jezu.wav', format="wav")
song.set_channels(1)
song = song[:44100]
np_samples = np.array(song.get_array_of_samples())[:44100]
print(len(np_samples))

frame_width = 441
num_frames = 100
s = time.time()
zcr_v, ste_v = zcr_ste(np_samples, frame_width, num_frames)
print(time.time() - s)
print(zcr_v[:60], '\n', ste_v[:60])

s = time.time()
zcr_v, ste_v = zcr_ste(np_samples, frame_width, num_frames)
print(time.time() - s)


stee = short_time_energy(np_samples, 441)
print(zcr_v[:60], '\n', ste_v[:60])

ehh =1
