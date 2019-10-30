import wave
import numpy as np
from matplotlib import pyplot as plt
import sys
import glob
from scipy.io import wavfile
from functions import zero_crossing_rate, short_time_energy

filepath = "yazoo.wav"
frame_width = 441

samplerate, wavedata = wavfile.read(filepath)

audio = {
    "samplerate": samplerate,
    "wavedata": wavedata,
    "number_of_samples": wavedata.shape[0],
    "audio_legth": int(wavedata.shape[0] / samplerate),
}
i = np.max(wavedata)
print("EHHHH")


