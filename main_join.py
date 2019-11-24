import glob
import pandas as pd
from functions import get_audio_features
from globals import DATASET_PATH, OUT_PATH
from globals import FRAME_WIDTH, ZCR_THRESHOLD, DS_NAME

dfs = []
for n in ('wav', 'anty', 'zet', 'fm'):
    df = pd.read_csv(OUT_PATH + n + '.csv', header=0)
    dfs.append(df)

for d in dfs[:-1]:
    df = pd.concat([df, d])

df.to_csv(OUT_PATH + 'zero.csv', index=None, header=True)