"""
Splits a wav file based on the contents of a csv file with the same name as the audio file (i.e. record.wav, record.csv)
The csv file should have three columns: minute, second, type (M or S).
Each row describes a change in the type of audio content.
Example:
0|  0,  0, M -> from 0m0s to 1m12s - music
1|  1, 12, S -> from 1m12s to 12m32s - not music
2| 12, 32, M -> from 12m32s  - music

"""
from pydub import AudioSegment
import csv
import os

DATASET_PATH = ''
FILE_PATH=''

for file in []:
    try:
        song = AudioSegment.from_wav(f'{FILE_PATH}{file}.wav' )
    except Exception:
        continue
    duration = song.duration_seconds
    print(f'Processing {file}.wav')
    type = []
    time_start = []
    time_stop = []
    with open(f'{file}.csv', newline='') as f:
        reader = csv.reader(f)
        for n, row in enumerate(reader):
            type.append(row[2])
            time_start.append(int(row[0])*60*1000 + int(row[1])*1000)
            if n > 0:
                time_stop.append(int(row[0])*60*1000 + int(row[1])*1000)

        time_stop.append(int(duration)*1000)

    for start, stop, type in zip(time_start, time_stop, type):
        try:
            os.mkdir(f'music_{file}')
            os.mkdir(f'speech_{file}')
        except:
            pass
        extract = song[start:stop]
        if type == 'M':
            folder = f'music_{file}'
        else:
            folder = f'speech_{file}'
        extract.export(f'{DATASET_PATH}{folder}/{type}_{str(start)}.wav', format="wav")
