import glob
import pandas as pd
from functions import get_audio_features
from globals import DATASET_PATH, OUT_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD, DS_NAME

dfs = []
# ZWYKLE
for n in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic'):
    df = pd.read_csv(OUT_PATH + n + '.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'hm.csv', index=None, header=True)

dfs = []
# ZWYKLE HMM MUZYKA
for n in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic'):
    df = pd.read_csv(OUT_PATH + n + 'm.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'hmm.csv', index=None, header=True)


dfs = []
# ZWYKLE HMM MOWA
for n in ('anty', 'jeden', 'dwa', 'trzy', 'olsztyn', 'zet', 'fm', 'classic'):
    df = pd.read_csv(OUT_PATH + n + 's.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'hms.csv', index=None, header=True)

# ======================================================================================
for n in ('wav', 'c', 'g', 't', 'f'):
    df = pd.read_csv(OUT_PATH + n + 'm.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'zero.csv', index=None, header=True)
for n in ('wav', 'c', 'g', 't', 'f'):
    df = pd.read_csv(OUT_PATH + n + 's.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'zeros.csv', index=None, header=True)



for n in ('wav', 'c', 'g', 't', 'f'):
    df = pd.read_csv(OUT_PATH + n + 'm.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'zerom.csv', index=None, header=True)

# for n in ('wav', 'c', 'g', 't', 'f'):
#     df = pd.read_csv(OUT_PATH + n + 's.csv', header=0)
#     dfs.append(df)
#
#
# for n in ('wav', 'c', 'g', 't', 'f'):
#     df = pd.read_csv(OUT_PATH + n + '.csv', header=0)
#     dfs.append(df)
#
# for d in dfs[:-1]:
#     df = pd.concat([df, d])
#
# df.to_csv(OUT_PATH + 'zero.csv', index=None, header=True)
#
# for d in dfs[:-1]:
#     df = pd.concat([df, d])
#
# df.to_csv(OUT_PATH + 'zeros.csv', index=None, header=True)