import numpy as np
from scipy.stats import moment
from scipy.io import wavfile
from statistics import stdev
from math import copysign


def timestamps(wavedata, frame_width, sample_rate):
    num_frames = int(np.ceil(len(wavedata) / frame_width))
    return (np.arange(0, num_frames - 1) * (frame_width / float(sample_rate)))


def zero_crossing_rate(wavedata, frame_width):
    num_frames = int(np.ceil(len(wavedata) / frame_width))
    zcr = []

    for i in range(0, num_frames - 1):
        start = i * frame_width
        stop = np.min([(start + frame_width - 1), len(wavedata)])

        zc = 1/(2*(frame_width-1)) *  np.count_nonzero(np.diff(np.sign(wavedata[start:stop])))
        zcr.append(zc)

    return np.asarray(zcr)


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


def zcr_diff_mean(zcr, zcr_mean):
    return ((zcr > zcr_mean).sum() - (zcr < zcr_mean).sum())/(len(zcr))


# def zcr_exceed_th(zcr, threshold):
#     return (zcr > threshold).sum() / (len(zcr))


def zcr_exceed_th(zcr_v, zcr_mean, control_coeff=1.2):
    def sgn(x):
        if x > 0:
            return 1
        else:
            return -1
    m = max(zcr_v)
    n = len(zcr_v)
    return np.sum([sgn(zcr - (m - control_coeff*zcr_mean)) + 1 for zcr in zcr_v]) / (2*n)



def zcr_third_central_moment(zcr, m=3):
    return moment(zcr, moment=m)


def zcr_std_of_fod(zcr):
    """ Standard deviation of the first order difference """
    return np.std(zcr)


def ste_mean(ste):
    return np.mean(ste)


def ste_mler(ste_v, control_coeff=0.1):
    """ Modified Low Energy Ratio """
    mean_ste = ste_mean(ste_v)
    if(mean_ste == 0):
        print('malutko')
    n = len(ste_v)

    def sgn(x):
        if x > 0:
            return 1
        else:
            return -1
    return np.sum([sgn(control_coeff*mean_ste - ste) + 1 for ste in ste_v]) / (2*n)


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
        'zcr_mean': zcr_mean,
        'ste_mler': ste_mler(ste),
    }
    return data

