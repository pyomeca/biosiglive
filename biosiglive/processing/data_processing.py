from scipy.signal import butter, lfilter, filtfilt, convolve
import numpy as np
import scipy.io as sio

try:
    from pyomeca import Analogs
    pyomeca_module = True
except ModuleNotFoundError:
    pyomeca_module = False


class GenericProcessing:
    def __init__(self):
        self.bpf_lcut = 10
        self.bpf_hcut = 425
        self.lpf_lcut = 5
        self.lp_butter_order = 4
        self.bp_butter_order = 4
        self.ma_win = 200

    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    @staticmethod
    def _butter_lowpass(lowcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, [low], btype="low")
        return b, a

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _butter_lowpass_filter(self, data, lowcut, fs, order=4):
        b, a = self._butter_lowpass(lowcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def _moving_average(data, w, empty_ma):
        for i in range(data.shape[0]):
            empty_ma[i, :] = convolve(data[i, :], w, mode="same", method="fft")
        return empty_ma

    @staticmethod
    def normalize_emg(emg_data, mvc_list):
        if len(mvc_list) == 0:
            raise RuntimeError("Please give a list of mvc to normalize the emg signal.")
        norm_emg = np.zeros((emg_data.shape[0], emg_data.shape[1]))
        for emg in range(emg_data.shape[0]):
            norm_emg[emg, :] = emg_data[emg, :] / mvc_list[emg]
        return norm_emg


class RealTimeProcessing(GenericProcessing):
    def __init__(self):
        self.emg_rate = 2000
        self.emg_win = 200
        self.ma_win = 20
        super().__init__()

    def process_emg_rt(self, raw_emg: np.ndarray,
                       emg_proc: np.ndarray,
                       emg_tmp: np.ndarray,
                       mvc_list: list,
                       norm_emg: bool = True,
                       lpf: bool = False):

        if self.ma_win > self.emg_win:
            raise RuntimeError(f"Moving average windows ({self.ma_win}) higher than emg windows ({self.emg_win}).")
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

        elif raw_emg.shape[1] < self.emg_win:
            emg_proc = np.zeros((emg_tmp.shape[0], self.emg_win))
            raw_emg = np.append(raw_emg, emg_tmp, axis=1)

        else:
            raw_emg = np.append(raw_emg[:, -self.emg_win + emg_sample:], emg_tmp, axis=1)
            emg_proc_tmp = abs(self._butter_bandpass_filter(raw_emg, self.bpf_lcut, self.bpf_hcut, self.emg_rate))

            if lpf is True:
                emg_lpf_tmp = self._butter_lowpass_filter(emg_proc_tmp, self.lpf_lcut, self.emg_rate, order=self.lp_butter_order)
                emg_lpf_tmp = emg_lpf_tmp[:, ::emg_sample]
                emg_proc = np.append(emg_proc[:, emg_sample:], emg_lpf_tmp[:, -emg_sample:], axis=1)

            else:
                average = np.median(emg_proc_tmp[:, -self.ma_win:], axis=1).reshape(-1, 1)
                emg_proc = np.append(emg_proc[:, 1:], average / quot, axis=1)
        return raw_emg, emg_proc

    @staticmethod
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
            raw_im = np.append(raw_im[:, :, -im_win + im_sample:], im_tmp, axis=2)
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

    @staticmethod
    def custom_processing(funct, raw_data, data_proc, data_tmp, *args, **kwargs):
        return funct(raw_data, data_proc, data_tmp, *args, **kwargs)


class OfflineProcessing(GenericProcessing):
    def __init__(self):
        super().__init__()

    def process_emg(self, data, frequency, pyomeca=False, ma=False):
        if pyomeca is True:
            if pyomeca_module is False:
                raise RuntimeError("Pyomeca module not found.")
            if ma is True:
                raise RuntimeError("Moving average not available with pyomeca.")
            emg = Analogs(data)
            emg_processed = (
                emg.meca.band_pass(order=self.bp_butter_order, cutoff=[self.bpf_lcut, self.bpf_hcut], freq=frequency)
                    .meca.abs()
                    .meca.low_pass(order=self.lp_butter_order, cutoff=self.lpf_lcut, freq=frequency)
            )
            emg_processed = emg_processed.values

        else:
            emg_processed = abs(self._butter_bandpass_filter(data, self.bpf_lcut, self.bpf_hcut, frequency))
            if ma is True:
                w = np.repeat(1, self.ma_win) / self.ma_win
                empty_ma = np.ndarray((data.shape[0], data.shape[1]))
                emg_processed = self._moving_average(emg_processed, w, empty_ma)
            else:
                emg_processed = self._butter_lowpass_filter(emg_processed, self.lpf_lcut, frequency, order=self.lp_butter_order)
        return emg_processed

    @staticmethod
    def custom_processing(funct, raw_data, *args, **kwargs):
        return funct(raw_data, *args, **kwargs)

    @staticmethod
    def compute_mvc(nb_muscles: int, mvc_trials: np.ndarray, window_size: int, file_name: str, mvc_list_max: np.ndarray):
        for i in range(nb_muscles):
            mvc_temp = -np.sort(-mvc_trials, axis=1)
            if i == 0:
                mvc_list_max = mvc_temp[:, :window_size]
            else:
                mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :window_size]), axis=1)
        mvc_list_max = -np.sort(-mvc_list_max, axis=1)[:, :window_size]
        mvc_list_max = np.median(mvc_list_max, axis=1)
        mat_content = sio.loadmat(file_name)
        mat_content["MVC_list_max"] = mvc_list_max


if __name__ == '__main__':
    pass
