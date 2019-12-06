import numpy as np
from scipy.stats import moment


def sgn(x):
    if x > 0:
        return 1
    else:
        return -1


def zcr_ste(samples: np.ndarray, frame_width: int, num_frames: int) -> (np.ndarray, np.ndarray):
    """ Funkcja zwracajaca wektory wartosci ZCR i STE

    :param samples: wektor wartosci ZCR w oknie
    :param frame_width: szerokosc ramki
    :param num_frames: ilosc ramek w oknie
    """
    chunks = np.array_split(samples, num_frames)

    f_zcr = lambda x: 1 / (2 * frame_width) * np.count_nonzero(np.diff(np.sign(x)))
    f_ste = lambda x: np.sum(np.square(x, dtype=np.int64))

    ste_v = np.array(list(map(f_ste, chunks)))
    zcr_v = np.array(list(map(f_zcr, chunks)))
    return zcr_v, ste_v


def zcr_diff_mean(zcr_v: np.ndarray, zcr_mean: float) -> float:
    """ Roznica pomiedzy liczba ramek z ZCR powyzej i ponizej sredniej

    :param zcr_v: wektor wartosci ZCR w oknie
    :param zcr_mean: srednia ZCR w oknie
    """
    return np.sum([sgn(zcr - zcr_mean) + 1 for zcr in zcr_v]) / (2*zcr_v.size)


def zcr_exceed_th(zcr_v: np.ndarray, zcr_mean: float, num_frames: int, control_coeff: float = 1.5) -> float:
    """ Liczba wystapien wartosci ZCR powyzej progu

    :param zcr_v: wektor wartosci ZCR w oknie
    :param zcr_mean: srednia ZCR w oknie
    :param num_frames: ilosc ramek w oknie
    :param control_coeff: wspolczynnik kontrolny parametru
    """
    m = max(zcr_v)
    n = num_frames
    return np.sum([sgn(zcr - (m - control_coeff*zcr_mean)) + 1 for zcr in zcr_v]) / (2*n)


def zcr_third_central_moment(zcr_v: np.ndarray) -> float:
    """ Trzeci moment centralny dla wartosci ZCR

    :param zcr_v: wektor wartosci ZCR w oknie
    """
    return moment(zcr_v, moment=3)*100


def zcr_std_of_fod(zcr_v: np.ndarray) -> float:
    """  Odchylenie standardowe pierwszej roznicy wstecznej dla kolejnych wartosci ZCR

    :param zcr_v: wektor wartosci ZCR w oknie
    """
    return np.std(np.diff(zcr_v))


def ste_mler(ste_v: np.ndarray, num_frames: int, control_coeff: float = 0.15) -> float:
    """ MLER

    :param ste_v: wektor wartosci STE w oknie
    :param num_frames: ilosc ramek w oknie
    :param control_coeff: wspolczynnik kontrolny parametru
    """
    ste_mean = np.mean(ste_v)
    n = num_frames
    return np.sum([sgn(control_coeff*ste_mean - ste) + 1 for ste in ste_v]) / (2*n)


def get_input_vector(samples: np.ndarray, frame_width: int, num_frames: int) -> np.ndarray:
    """ Wektor wejsciowy parametrow klasyfikatora

    :param samples: probki sygnalu
    :param frame_width: szerokosc ramki
    :param num_frames: ilosc ramek w oknie
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