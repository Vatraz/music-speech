import os
from globals import DATASET_PATH


def check_directories(dir_pattern):
    music_dir = DATASET_PATH + dir_pattern.format('music')
    speech_dir = DATASET_PATH + dir_pattern.format('speech')
    for directory in (music_dir, speech_dir):
        if not os.path.exists(directory):
            os.makedirs(directory)
    return  music_dir, speech_dir