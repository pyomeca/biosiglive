import numpy as np
from scipy.signal import butter, lfilter, filtfilt, convolve
import pickle
from time import time

try:
    from casadi import MX, Function, horzcat

    casadi_package = True
except ModuleNotFoundError:
    casadi_package = False

try:
    from pyomeca import Analogs

    pyomeca_module = True
except ModuleNotFoundError:
    pyomeca_module = False


def process_emg(data, frequency, bpf_lcut=10, bpf_hcut=425, lpf_lcut=5, order=4, ma_win=200, pyomeca=False, ma=False):
    if pyomeca is True:
        if pyomeca_module is False:
            raise RuntimeError("Pyomeca module not found.")
        if ma is True:
            raise RuntimeError("Moving average not available with pyomeca.")
        emg = Analogs(data)
        emg_processed = (
            emg.meca.band_pass(order=order, cutoff=[bpf_lcut, bpf_hcut], freq=frequency)
            .meca.abs()
            .meca.low_pass(order=order, cutoff=lpf_lcut, freq=frequency)
        )
        emg_processed = emg_processed.values

    else:
        emg_processed = abs(butter_bandpass_filter(data, bpf_lcut, bpf_hcut, frequency))
        if ma is True:
            w = np.repeat(1, ma_win) / ma_win
            empty_ma = np.ndarray((data.shape[0], data.shape[1]))
            emg_processed = moving_average(emg_processed, w, empty_ma)
        else:
            emg_processed = butter_lowpass_filter(emg_processed, lpf_lcut, frequency, order=order)
    return emg_processed


def normalize_emg(emg_data, mvc_list):
    if len(mvc_list) == 0:
        raise RuntimeError("Please give a list of mvc to normalize the emg signal.")
    norm_emg = np.zeros((emg_data.shape[0], emg_data.shape[1]))
    for emg in range(emg_data.shape[0]):
        norm_emg[emg, :] = emg_data[emg, :] / mvc_list[emg]
    return norm_emg


def process_imu(
    im_proc,
    raw_im,
    im_tmp,
    im_win,
    im_sample,
    ma_win,
    accel=False,
    squared=False,
    norm_min_bound=None,
    norm_max_bound=None,
):

    if len(raw_im) == 0:
        if squared is not True:
            im_proc = np.zeros((im_tmp.shape[0], im_tmp.shape[1], 1))
        else:
            im_proc = np.zeros((im_tmp.shape[0], 1))
        raw_im = im_tmp

    elif raw_im.shape[2] < im_win:
        if squared is not True:
            im_proc = np.zeros((im_tmp.shape[0], im_tmp.shape[1], im_win))
        else:
            im_proc = np.zeros((im_tmp.shape[0], im_win))
        raw_im = np.append(raw_im, im_tmp, axis=2)

    else:
        raw_im = np.append(raw_im[:, :, -im_win + im_sample :], im_tmp, axis=2)
        im_proc_tmp = raw_im
        average = np.mean(im_proc_tmp[:, :, -ma_win:], axis=2).reshape(-1, 3, 1)
        if squared:
            if accel:
                average = abs(np.linalg.norm(average, axis=1) - 9.81)
            else:
                average = np.linalg.norm(average, axis=1)

        if len(average.shape) == 3:
            if norm_min_bound or norm_max_bound:
                for i in range(raw_im.shape[0]):
                    for j in range(raw_im.shape[1]):
                        if average[i, j, :] < 0:
                            average[i, j, :] = average[i, j, :] / abs(norm_min_bound)
                        elif average[i, j, :] >= 0:
                            average[i, j, :] = average[i, j, :] / norm_max_bound
            im_proc = np.append(im_proc[:, :, 1:], average, axis=2)

        else:
            if norm_min_bound or norm_max_bound:
                for i in range(raw_im.shape[0]):
                    for j in range(raw_im.shape[1]):
                        if average[i, :] < 0:
                            average[i, :] = average[i, :] / abs(norm_min_bound)
                        elif average[i, :] >= 0:
                            average[i, :] = average[i, :] / norm_max_bound
            im_proc = np.append(im_proc[:, 1:], average, axis=1)

    return raw_im, im_proc


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_lowpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, [low], btype="low")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass_filter(data, lowcut, fs, order=4):
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def moving_average(data, w, empty_ma):
    for i in range(data.shape[0]):
        empty_ma[i, :] = convolve(data[i, :], w, mode="same", method="fft")
    return empty_ma


def process_emg_rt(raw_emg, emg_proc, emg_tmp, mvc_list, ma_win, emg_win=2000, emg_freq=2000, norm_emg=True, lpf=False):
    if ma_win > emg_win:
        raise RuntimeError(f"Moving average windows ({ma_win}) higher than emg windows ({emg_win}).")
    emg_sample = emg_tmp.shape[1]
    if norm_emg is True:
        if isinstance(mvc_list, np.ndarray) is True:
            if len(mvc_list.shape) == 1:
                quot = mvc_list.reshape(-1, 1)
            else:
                quot = mvc_list
        else:
            quot = np.array(mvc_list).reshape(-1, 1)
    else:
        quot = [1]

    if len(raw_emg) == 0:
        emg_proc = np.zeros((emg_tmp.shape[0], 1))
        raw_emg = emg_tmp

    elif raw_emg.shape[1] < emg_win:
        emg_proc = np.zeros((emg_tmp.shape[0], emg_win))
        raw_emg = np.append(raw_emg, emg_tmp, axis=1)

    else:
        raw_emg = np.append(raw_emg[:, -emg_win + emg_sample :], emg_tmp, axis=1)
        emg_proc_tmp = abs(butter_bandpass_filter(raw_emg, 10, 425, emg_freq))

        if lpf is True:
            emg_lpf_tmp = butter_lowpass_filter(emg_proc_tmp, 5, emg_freq, order=4)
            emg_lpf_tmp = emg_lpf_tmp[:, ::emg_sample]
            emg_proc = np.append(emg_proc[:, emg_sample:], emg_lpf_tmp[:, -emg_sample:], axis=1)

        else:
            average = np.mean(emg_proc_tmp[:, -ma_win:], axis=1).reshape(-1, 1)
            emg_proc = np.append(emg_proc[:, 1:], average / quot, axis=1)
    return raw_emg, emg_proc


def add_data_to_pickle(data_dict, data_path):
    with open(data_path, "ab") as outf:
        pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


def read_data(filename):
    data = {}
    with open(filename, "rb") as file:
        count = 0
        while True:
            try:
                data_tmp = pickle.load(file)
                for key in data_tmp.keys():
                    if count == 0:
                        if isinstance(data_tmp[key], (int, float, str)) is True:
                            data[key] = [data_tmp[key]]
                        else:
                            data[key] = data_tmp[key]
                    else:
                        if isinstance(data_tmp[key], (int, float, str)) is True:
                            if key != "nb_data_of_interest" and key != "init_w_kalman" and key != "read_freq":
                                data[key].append(data_tmp[key])
                        elif len(data_tmp[key].shape) == 2:
                            data[key] = np.append(data[key], data_tmp[key], axis=1)
                        elif len(data_tmp[key].shape) == 3:
                            data[key] = np.append(data[key], data_tmp[key], axis=2)
                if count == 0:
                    count += 1
            except EOFError:
                break
    return data


def add_to_timeline(init_time, timeline):
    return timeline.append(time() - init_time)


# Use biorbd function for inverse kinematics
def markers_fun(biorbd_model, q=None, eigen_backend=False):
    if eigen_backend:
        return [biorbd_model.markers(q)[i].to_array() for i in range(biorbd_model.nbMarkers())]
    else:
        qMX = MX.sym("qMX", biorbd_model.nbQ())
        return Function(
            "markers",
            [qMX],
            [horzcat(*[biorbd_model.markers(qMX)[i].to_mx() for i in range(biorbd_model.nbMarkers())])],
        )
