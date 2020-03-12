from scipy.stats import moment
import numpy as np


def sgn(x):
    if x > 0:
        return 1
    else:
        return -1


def zcr_ste(samples: np.ndarray, frame_width: int, num_frames: int) -> (np.ndarray, np.ndarray):
    """
    Returns ZCR and STE values

    :param samples: vector of ZCR values in the window
    :param frame_width: frame width
    :param num_frames: number of frames in the window
    """
    chunks = np.array_split(samples, num_frames)

    f_zcr = lambda x: 1 / (2 * frame_width) * np.count_nonzero(np.diff(np.sign(x)))
    f_ste = lambda x: np.sum(np.square(x, dtype=np.int64))

    ste_v = np.array(list(map(f_ste, chunks)))
    zcr_v = np.array(list(map(f_zcr, chunks)))
    return zcr_v, ste_v


def zcr_diff_mean(zcr_v: np.ndarray, zcr_mean: float) -> float:
    """ Difference between frames with ZCR value above and below the mean

    :param zcr_v: vector of ZCR values in the window
    :param zcr_mean: average ZCR in the window
    """
    return np.sum([sgn(zcr - zcr_mean) + 1 for zcr in zcr_v]) / (2*zcr_v.size)


def zcr_exceed_th(zcr_v: np.ndarray, zcr_mean: float, num_frames: int, control_coeff: float = 1.5) -> float:
    """ Number of frames with ZCR value exceeding threshold

    :param zcr_v: vector of ZCR values in the window
    :param zcr_mean: average value of ZCR in the window
    :param num_frames: total number of frames in the window
    :param control_coeff: control coefficient
    """
    m = max(zcr_v)
    n = num_frames
    return np.sum([sgn(zcr - (m - control_coeff*zcr_mean)) + 1 for zcr in zcr_v]) / (2*n)


def zcr_third_central_moment(zcr_v: np.ndarray) -> float:
    """ Third central moment about the mean

    :param zcr_v: vector of ZCR values in the window
    """
    return moment(zcr_v, moment=3)*100


def zcr_std_of_fod(zcr_v: np.ndarray) -> float:
    """
    Standard deviation of first order difference of ZCR

    :param zcr_v: vector of ZCR values in the window
    """
    return np.std(np.diff(zcr_v))


def ste_mler(ste_v: np.ndarray, num_frames: int, control_coeff: float = 0.15) -> float:
    """ MLER - Modified Low Energy Ratio

    :param ste_v: vector of STE values in the window
    :param num_frames: number of frames in the window
    :param control_coeff: control coefficient
    """
    ste_mean = np.mean(ste_v)
    n = num_frames
    return np.sum([sgn(control_coeff*ste_mean - ste) + 1 for ste in ste_v]) / (2*n)


def get_input_vector(samples: np.ndarray, frame_width: int, num_frames: int) -> np.ndarray:
    """
    Returns the vector of classifiers input values

    :param samples: samples
    :param frame_width: frame width
    :param num_frames: number of frames in the window
    """
    zcr_v, ste_v = zcr_ste(samples, frame_width, num_frames)
    zcr_mean = zcr_v.mean()
    input_vector = [
        zcr_diff_mean(zcr_v, zcr_mean),
        zcr_third_central_moment(zcr_v),
        zcr_exceed_th(zcr_v, zcr_mean, num_frames),
        zcr_std_of_fod(zcr_v),
        ste_mler(ste_v, num_frames),
    ]
    return np.array([input_vector])
