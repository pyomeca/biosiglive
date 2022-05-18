"""
This file is part of biosiglive. it contains the functions for data processing (offline and in real-time).
"""

from scipy.signal import butter, lfilter, filtfilt, convolve
import numpy as np
import scipy.io as sio
import os
from typing import Union

try:
    from pyomeca import Analogs
    pyomeca_module = True
except ModuleNotFoundError:
    pyomeca_module = False


class GenericProcessing:
    def __init__(self):
        """
        Initialize the class.
        """
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
    def _moving_average(data: np.ndarray, w, empty_ma):
        for i in range(data.shape[0]):
            empty_ma[i, :] = convolve(data[i, :], w, mode="same", method="fft")
        return empty_ma

    @staticmethod
    def normalize_emg(emg_data: np.ndarray, mvc_list: list):
        """
        Normalize EMG data.

        Parameters
        ----------
        emg_data : numpy.ndarray
            EMG data.
        mvc_list : list
            List of MVC values.

        Returns
        -------
        numpy.ndarray
            Normalized EMG data.
        """

        if len(mvc_list) == 0:
            raise RuntimeError("Please give a list of mvc to normalize the emg signal.")
        norm_emg = np.zeros((emg_data.shape[0], emg_data.shape[1]))
        for emg in range(emg_data.shape[0]):
            norm_emg[emg, :] = emg_data[emg, :] / mvc_list[emg]
        return norm_emg


class RealTimeProcessing(GenericProcessing):
    def __init__(self):
        """
        Initialize the class for real time processing.
        """
        self.emg_rate = 2000
        self.emg_win = 200
        self.ma_win = 200
        super().__init__()

    def process_emg(self,
                    raw_emg: np.ndarray,
                    emg_proc: np.ndarray,
                    emg_tmp: np.ndarray,
                    mvc_list: Union[list, tuple],
                    norm_emg: bool = True,
                    lpf: bool = False):
        """
        Process EMG data in real-time.
        Parameters
        ----------
        raw_emg : numpy.ndarray
            Raw EMG data. (nb_emg, emg_win * emg_sample)
        emg_proc : numpy.ndarray
            Last processed EMG data. (nb_emg, emg_win)
        emg_tmp : numpy.ndarray
            Temporary EMG data (nb_emg, emg_sample).
        mvc_list : list
            MVC values.
        norm_emg : bool
            Normalize EMG data.
        lpf : bool
            Apply low-pass filter.
        Returns
        -------
        tuple
            raw and processed EMG data.

        """
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
        """
        Process IMU data in real-time.
        Parameters
        ----------
        im_proc : numpy.ndarray
            Last processed IMU data. (nb_imu, im_win)
        raw_im : numpy.ndarray
            Raw IMU data. (nb_imu, im_win * im_sample)
        im_tmp : numpy.ndarray
            Temporary IMU data (nb_imu, im_sample).
        im_win : int
            IMU window size.
        im_sample : int
            IMU sample size.
        ma_win : int
            Moving average window size.
        accel : bool
            If current data is acceleration data to adapt the processing.
        squared : bool
            Apply squared.
        norm_min_bound : tuple
            Normalization minimum bound.
        norm_max_bound : tuple
            Normalization maximum bound.

        Returns
        -------
        tuple
            raw and processed IMU data.
        """

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
    def get_peaks(new_sample: np.ndarray,
                  signal: np.ndarray,
                  signal_proc: np.ndarray,
                  threshold: float,
                  chanel_idx: Union[int, list] = None,
                  nb_min_frame: float = 2000,
                  is_one = None,
                  min_peaks_interval = 0
                  ):
        """
        Allow to get the number of peaks for an analog signal (to get cadence from treadmill for instance).
        Parameters
        ----------
        new_sample
        signal
        threshold
        chanel
        window_len
        rate
        nb_min_frame

        Returns
        -------

        """
        nb_peaks = []
        # is_one = False
        if len(new_sample.shape) == 1:
            new_sample = np.expand_dims(new_sample, 0)
        sample_proc = np.copy(new_sample)

        for i in range(new_sample.shape[0]):
            for j in range(new_sample.shape[1]):
                if new_sample[i, j] < threshold:
                    sample_proc[i, j] = 0
                    is_one[i] = False
                elif new_sample[i, j] >= threshold:
                    if not is_one[i]:
                        if np.count_nonzero(new_sample == 1):
                            pass
                        if len(signal_proc) != 0 and np.count_nonzero(signal_proc[:, -min_peaks_interval:] == 1):
                            pass
                        else:
                            sample_proc[i, j] = 1
                            is_one[i] = True
                    else:
                        sample_proc[i, j] = 0

        if len(signal) == 0:
            signal = new_sample
            signal_proc = sample_proc
            nb_peaks = np.zeros((1, 1))

        elif signal.shape[1] < nb_min_frame:
            signal = np.append(signal, new_sample, axis=1)
            signal_proc = np.append(signal_proc, sample_proc, axis=1)
            nb_peaks = np.zeros((1, 1))

        else:
            signal = np.append(signal[:, -nb_min_frame + new_sample.shape[1]:], new_sample, axis=1)
            signal_proc = np.append(signal_proc[:, -nb_min_frame + new_sample.shape[1]:], sample_proc, axis=1)

        if chanel_idx:
            signal = signal[chanel_idx, :]
            signal_proc = signal_proc[chanel_idx, :]

        if isinstance(nb_peaks, list):
            nb_peaks.append(np.count_nonzero(signal_proc[:, :] == 1))
        return nb_peaks, signal_proc, signal, is_one

    @staticmethod
    def custom_processing(funct, raw_data, data_proc, data_tmp, *args, **kwargs):
        return funct(raw_data, data_proc, data_tmp, *args, **kwargs)


class OfflineProcessing(GenericProcessing):
    def __init__(self):
        """
        Offline processing.
        """
        super().__init__()

    def process_emg(self, data, frequency, pyomeca=False, ma=False):
        """
        Process EMG data.
        Parameters
        ----------
        data : numpy.ndarray
            EMG data.
        frequency : int
            EMG data frequency.
        pyomeca : bool
            If true, use low pass filter from pyomeca.
        ma : bool
            If true, apply moving average.
        Returns
        -------
        numpy.ndarray
            Processed EMG data.
        """
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
    def compute_mvc(nb_muscles: int,
                    mvc_trials: np.ndarray,
                    window_size: int,
                    mvc_list_max: np.ndarray,
                    tmp_file: str = None,
                    output_file: str = None,
                    save: bool = False):
        """
        Compute MVC from several mvc_trials.

        Parameters
        ----------
        nb_muscles : int
            Number of muscles.
        mvc_trials : numpy.ndarray
            EMG data for all trials.
        window_size : int
            Size of the window.
        mvc_list_max : numpy.ndarray
            List of maximum MVC for each muscle.
        tmp_file : str
            Name of the temporary file.
        output_file : str
            Name of the output file.
        save : bool
            If true, save the results.
        Returns
        -------
        list
            MVC for each muscle.

        """
        for i in range(nb_muscles):
            mvc_temp = -np.sort(-mvc_trials, axis=1)
            if i == 0:
                mvc_list_max = mvc_temp[:, :window_size]
            else:
                mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :window_size]), axis=1)
        mvc_list_max = -np.sort(-mvc_list_max, axis=1)[:, :window_size]
        mvc_list_max = np.median(mvc_list_max, axis=1)

        if tmp_file:
            mat_content = sio.loadmat(tmp_file)
            mat_content["MVC_list_max"] = mvc_list_max
        else:
            mat_content = {"MVC_list_max": mvc_list_max, "MVC_trials": mvc_trials}

        if save:
            sio.savemat(output_file, mat_content)
        os.remove(tmp_file)
        return mvc_list_max