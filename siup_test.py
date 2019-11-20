from pydub import AudioSegment
from functions import *
from scipy.io import wavfile
import sys

import numpy as np

song = AudioSegment.from_file('hmm5.wav', format="wav")
np_samples = np.array(song.get_array_of_samples())

frame_width = 450
num_frames = int(np.ceil(np_samples.size / frame_width))
zcr_v, ste_v = zcr_ste(np_samples, frame_width, num_frames)
zcr_mean = np.mean(zcr_v)
ste_mler(ste_v, control_coeff=0.1)
zcr_diff_mean(zcr_v, zcr_mean)
zcr_exceed_th(zcr_v, zcr_mean, control_coeff=1.2)
zcr_third_central_moment(zcr_v)
zcr_std_of_fod(zcr_v)

ehh =1
