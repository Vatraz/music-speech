import glob
import librosa
from dataset_utils import check_directories
from globals import DATASET_PATH


speech_files, music_files = glob.glob(DATASET_PATH + "sp/*.wav"), \
                            glob.glob(DATASET_PATH + "ms/*.wav")


def resample(audio_files, output_directory, out_samplerate):
    counter, fin = 0, len(audio_files)
    for filepath in audio_files:
        if counter % 5 == 0:
            print(counter, '/', fin)
        counter += 1
        y, sr = librosa.load(filepath, sr=out_samplerate)  # Downsample
        if y.shape[0] == 2:
            y = librosa.to_mono(y)
        # new_filename = f'.'.join(filepath.split('/')[-1].split('.'))
        # path = f'{output_directory}/{new_filename}'
        new_filename = f'.'.join(filepath.split('\\')[-1].split('.'))
        path = f'{output_directory}\\{new_filename}'
        librosa.output.write_wav(path, y, sr)


out_samplerate = 16000  # Hz

music_dir, speech_dir = check_directories('{}_' + 'test' + '_16k')
resample(speech_files, speech_dir, out_samplerate)
resample(music_files, music_dir, out_samplerate)

