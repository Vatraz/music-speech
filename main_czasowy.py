import glob

from functions import get_audio_features
from globals import DATASET_PATH, FREQUENCY, FRAME_WIDTH
from numpy import loadtxt
from globals import OUT_PATH, DS_NAME
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.python.keras import models
import os
freq = FREQUENCY
# model = models.load_model('siup.hdf5')
for tup in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic'):
    s = sorted(glob.glob(DATASET_PATH + f'speech_' + tup + '_25_22k/*.wav'), key=lambda x: int(x.split('S_')[-1][:-4]))
    s_f = sorted(glob.glob(DATASET_PATH + f'speech_' + tup + '_22k/*.wav'), key=lambda x: int(x.split('S_')[-1][:-4]))
    m = sorted(glob.glob(DATASET_PATH + f'music_' + tup + '_25_22k/*.wav'), key=lambda x: int(x.split('M_')[-1][:-4]))
    m_f = sorted(glob.glob(DATASET_PATH + f'music_' + tup + '_22k/*.wav'), key=lambda x: int(x.split('M_')[-1][:-4]))
    m_f = [file[:-4] for file in m_f]
    s_f = [file[:-4] for file in s_f]


    # print(m_f)

    if s[0].endswith('_00.wav'):
        p = s # pierwsze, startuje mowa
        p_f = s_f
        d = m # drugie
        d_f = m_f
        typ_p = 0
        typ_d = 1
    else:
        d = s # drugie, bo startuje muzyka
        d_f = s_f
        p = m # pierwsze
        p_f = m_f
        typ_d = 0
        typ_p = 1

    files = []
    typy = []
    index = 0
    while True:
        try:
            start = p_f[index]
            start = '_25_22k'.join(start.split('_22k'))
            for plik_do in [x for x in p if x.startswith(start)]:
                files.append(plik_do)

                typy.append(typ_p)

        except:
            break
        try:
            start = d_f[index]
            start = '_25_22k'.join(start.split('_22k'))
            for plik_do in [x for x in d if x.startswith(start)]:
                files.append(plik_do)
                typy.append(typ_d)

        except:
            break
        index += 1
    data = [get_audio_features(file, FRAME_WIDTH, sound_type=1) for file in files]
    print('tup ', tup)
    df = pd.DataFrame(data)
    dft = pd.DataFrame({'typ': typy})
    df['type'] = dft['typ']
    df.to_csv('czas/' + tup + '.csv', index=None, header=True)



