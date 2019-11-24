import numpy as np
from scipy.stats import moment
from scipy.io import wavfile
from statistics import stdev
from math import copysign


def sgn(x):
    if x > 0:
        return 1
    else:
        return -1

def timestamps(wavedata, frame_width, sample_rate):
    num_frames = int(np.ceil(len(wavedata) / frame_width))
    return (np.arange(0, num_frames - 1) * (frame_width / float(sample_rate)))


def zcr_ste(samples: np.ndarray, frame_width: int, num_frames: int):
    zcr_v, ste_v = [], []
    for i in range(num_frames):
        frame = samples[i*frame_width:(i+1)*frame_width]
        zcr = 1/(2*frame_width) * np.count_nonzero(np.diff(np.sign(frame)))
        zcr_v.append(zcr)
        ste = np.sum(np.square(frame, dtype=np.int64))
        if ste < 0:
            print("rhh")
        ste_v.append(ste)
    return np.array(zcr_v), np.array(ste_v)


def zero_crossing_rate(wavedata, frame_width):
    num_frames = int(np.ceil(len(wavedata) / frame_width))
    zcr = []

    for i in range(0, num_frames - 1):
        start = i * frame_width
        stop = np.min([(start + frame_width - 1), len(wavedata)])

        zc = 1/(2*(frame_width-1)) *  np.count_nonzero(np.diff(np.sign(wavedata[start:stop])))

        zcr.append(zc)

    return np.asarray(zcr)


def zero_crossing_rate2(wavedata, frame_width, num_frames):
    zcr = []

    for i in range(num_frames):
        frame = wavedata[i*frame_width:(i+1)*frame_width]
        zc = 1/(2*frame_width) * np.count_nonzero(np.diff(np.sign(frame)))
        zcr.append(zc)

    return np.asarray(zcr)


def short_time_energy2(wavedata, frame_width, num_frames):
    ste = []

    for i in range(num_frames):
        start = i * frame_width
        stop = np.min([(start + frame_width - 1), len(wavedata)])
        energy = np.sum(np.square(np.asarray(wavedata[start:stop])))
        ste.append(energy)
        if energy < 0:
            print("OHO")
    return np.asarray(ste)


def short_time_energy(wavedata, frame_width):
    num_frames = int(np.ceil(len(wavedata) / frame_width))

    ste = []

    for i in range(num_frames - 1):
        start = i * frame_width
        stop = np.min([(start + frame_width - 1), len(wavedata)])
        energy = np.sum(np.square(np.asarray(wavedata[start:stop])))
        ste.append(energy)
        if energy < 0:
            print("OHO")
    return np.asarray(ste)


def zcr_mean_value(zcr):
    return np.mean(zcr)


# def zcr_diff_mean1(zcr, zcr_mean):
#     return ((zcr > zcr_mean).sum() - (zcr < zcr_mean).sum())/(len(zcr))


def zcr_diff_mean(zcr_v, zcr_mean):
    return np.sum([sgn(zcr - zcr_mean) + 1 for zcr in zcr_v]) / (2*zcr_v.size)

# def zcr_exceed_th(zcr, threshold):
#     return (zcr > threshold).sum() / (len(zcr))


def zcr_exceed_th(zcr_v, zcr_mean, control_coeff=1.3):
    m = max(zcr_v)
    n = len(zcr_v)
    return np.sum([sgn(zcr - (m - control_coeff*zcr_mean)) + 1 for zcr in zcr_v]) / (2*n)


def zcr_third_central_moment(zcr_v):
    return moment(zcr_v, moment=3)


def zcr_std_of_fod(zcr_v):
    """ Standard deviation of the first order difference """
    return np.std(np.diff(zcr_v))


def ste_mean(ste):
    return np.mean(ste)

def ste_mler(ste_v, control_coeff=0.12):
    ste_mean = np.mean(ste_v)
    n = len(ste_v)

    return np.sum([sgn(control_coeff*ste_mean - ste) + 1 for ste in ste_v]) / (2*n)


def read_audio_file(filepath, frame_width, zcr_threshold):
    samplerate, wavedata = wavfile.read(filepath)

    data = {
        'filepath': filepath,
        "samplerate": samplerate,
        "wavedata": wavedata,
        "number_of_samples": wavedata.shape[0],
        "audio_legth": int(wavedata.shape[0] / samplerate),
        'timestamps': timestamps(wavedata, frame_width, samplerate),
        'zcr': zero_crossing_rate(wavedata, frame_width),
        'ste': short_time_energy(wavedata, frame_width)
    }

    data['zcr_mean'] = zcr_mean_value(data['zcr'])
    data['zcr_diff_mean'] = zcr_diff_mean(data['zcr'], data['zcr_mean'])
    data['zcr_exceed_th'] = zcr_exceed_th(data['zcr'], zcr_threshold)
    data['zcr_third_central_moment'] = zcr_third_central_moment(data['zcr'])
    data['zcr_std_of_fod'] = zcr_std_of_fod(data['zcr'])

    data['ste_mean'] = ste_mean(data['ste'])
    data['ste_mler'] = ste_mler(data['ste'])

    return data


def get_audio_features(filepath, frame_width, sound_type):
    samplerate, wavedata = wavfile.read(filepath)
    zcr = zero_crossing_rate(wavedata, frame_width)
    ste = short_time_energy(wavedata, frame_width)
    zcr_mean = zcr_mean_value(zcr)
    data = {
        'type': sound_type,
        'zcr_diff_mean': zcr_diff_mean(zcr, zcr_mean),
        'zcr_third_central_moment': zcr_third_central_moment(zcr),
        'zcr_exceed_th': zcr_exceed_th(zcr, zcr_mean),
        'zcr_std_of_fod': zcr_std_of_fod(zcr),
        'ste_mler': ste_mler(ste),
        # 'zcr_mean': zcr_mean,
    }
    return data

