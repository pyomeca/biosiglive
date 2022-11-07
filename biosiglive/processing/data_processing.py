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


# TODO add a calibrate fucntion for calibration matrix
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
        self.moving_average_windows = 200
        self.data_rate = 2000

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

    def butter_lowpass_filter(self, data, lowcut, fs, order=4):
        b, a = self._butter_lowpass(lowcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def _moving_average(data: np.ndarray, w, empty_ma):
        for i in range(data.shape[0]):
            empty_ma[i, :] = convolve(data[i, :], w, mode="same", method="fft")
        return empty_ma

    @staticmethod
    def center(emg_data, center_value=None):
        center_value = center_value if center_value else emg_data.mean(axis=1)
        emg_centered = np.copy(emg_data)
        for i in range(emg_data.shape[0]):
            emg_centered[i, :] = emg_data[i, :] - center_value[i]
        return emg_centered

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

    @staticmethod
    def calibration_matrix(data: np.ndarray, matrix: np.ndarray):
        """
        Apply a calibration matrix to the data.
        Parameters
        ----------
        data: numpy.ndarray
            Data to calibrate.
        matrix: numpy.ndarray
            Calibration matrix.

        Returns
        -------

        """
        return np.dot(data, matrix)

    def _process_emg(self, data: np.ndarray,
                    mvc_list: Union[list, tuple] = None,
                    band_pass_filter=True,
                    low_pass_filter=False,
                    moving_average=True,
                    centering=True,
                    absolute_value=True,
                    normalization=False):
        """
        Process EMG data.
        Parameters
        ----------
        data : numpy.ndarray
            EMG data.
        mvc_list : list
            List of MVC values.
        band_pass_filter : bool
            Apply band pass filter.
        low_pass_filter : bool
            Apply low pass filter.
        moving_average : bool
            Apply moving average.
        centering : bool
            Apply centering.
        absolute_value : bool
            Apply absolute value.
        normalization : bool
            Apply normalization.
        Returns
        -------
        numpy.ndarray
            Processed EMG data.
        """
        data_proc = data
        if band_pass_filter:
            data_proc = self._butter_bandpass_filter(data, self.bpf_lcut, self.bpf_hcut, self.data_rate)
        if centering:
            data_proc = self.center(data_proc)
        if absolute_value:
            data_proc = np.abs(data_proc)
        if low_pass_filter and moving_average:
            raise RuntimeError("Please choose between low-pass filter and moving average.")
        if low_pass_filter:
            data_proc = self.butter_lowpass_filter(
                data_proc, self.lpf_lcut, self.data_rate, order=self.lp_butter_order
            )
        else:
            w = np.repeat(1, self.moving_average_windows) / self.moving_average_windows
            empty_ma = np.ndarray((data.shape[0], data.shape[1]))
            data_proc = self._moving_average(data_proc, w, empty_ma)

        if normalization:
            data_proc = self.normalize_emg(data_proc, mvc_list)

        return data_proc


class RealTimeProcessing(GenericProcessing):
    def __init__(self, data_rate: int = 2000, processing_windows: int = 2000, moving_average_window: int = 200):
        """
        Initialize the class for real time processing.
        """
        super().__init__()
        self.data_rate = data_rate
        self.processing_window = processing_windows
        self.ma_win = moving_average_window
        self.raw_data_buffer = []
        self.processed_data_buffer = []

    def process_emg(
        self,
        emg_data: np.ndarray,
        mvc_list: Union[list, tuple] = None,
        band_pass_filter = True,
        low_pass_filter = False,
        moving_average = True,
        centering = True,
        absolute_value = True,
        normalization = False,
        ):
        """
        Process EMG data in real-time.
        Parameters
        ----------
        emg_data : numpy.ndarray
            Temporary EMG data (nb_emg, emg_sample).
        mvc_list : list
            MVC values.
        Returns
        -------
        tuple
            raw and processed EMG data.

        """
        if self.ma_win > self.processing_window:
            raise RuntimeError(f"Moving average windows ({self.ma_win}) higher than emg windows ({self.processing_window}).")
        emg_sample = emg_data.shape[1]

        if normalization:
            if not mvc_list:
                raise RuntimeError("Please give a list of mvc to normalize the emg signal.")
            if isinstance(mvc_list, np.ndarray) is True:
                if len(mvc_list.shape) == 1:
                    quot = mvc_list.reshape(-1, 1)
                else:
                    quot = mvc_list
            else:
                quot = np.array(mvc_list).reshape(-1, 1)
        else:
            quot = [1]

        if len(self.raw_data_buffer) == 0:
            self.processed_data_buffer = np.zeros((emg_data.shape[0], 1))
            self.raw_data_buffer = emg_data

        elif self.raw_data_buffer.shape[1] < self.processing_window:
            self.raw_data_buffer = np.append(self.raw_data_buffer, emg_data, axis=1)
            self.processed_data_buffer = np.zeros((emg_data.shape[0], self.raw_data_buffer.shape[1]))

        else:
            self.raw_data_buffer = np.append(self.raw_data_buffer[:, -self.processing_window + emg_sample :], emg_data, axis=1)
            emg_proc_tmp =  self._process_emg(self.raw_data_buffer, band_pass_filter=band_pass_filter,
                                                     centering=centering,
                                                     absolute_value=absolute_value,
                                                     low_pass_filter=False,
                                                     moving_average=False,
                                                     normalization=False
                                                     )
            if low_pass_filter and moving_average:
                raise RuntimeError("Please choose between low-pass filter and moving average.")
            if low_pass_filter:
                emg_lpf_tmp = self.butter_lowpass_filter(
                    emg_proc_tmp, self.lpf_lcut, self.data_rate, order=self.lp_butter_order
                )
                emg_lpf_tmp = emg_lpf_tmp[:, ::emg_sample]
                self.processed_data_buffer = np.append(self.processed_data_buffer[:, emg_sample:], emg_lpf_tmp[:, -emg_sample:], axis=1)
            else:
                average = np.median(emg_proc_tmp[:, -self.ma_win:], axis=1).reshape(-1, 1)
                self.processed_data_buffer = np.append(self.processed_data_buffer[:, 1:], average / quot, axis=1)
        return self.processed_data_buffer

    def process_imu(self,
        im_data: np.ndarray,
        accel: bool =False,
        squared: bool =False,
        norm_min_bound:int=None,
        norm_max_bound: int=None,
    ):
        """
        Process IMU data in real-time.
        Parameters
        ----------
        im_data : numpy.ndarray
            Temporary IMU data (nb_imu, im_sample).
        accel : bool
            If current data is acceleration data to adapt the processing.
        squared : bool
            Apply squared.
        norm_min_bound : int
            Normalization minimum bound.
        norm_max_bound : int
            Normalization maximum bound.

        Returns
        -------
        tuple
            raw and processed IMU data.
        """

        if len(self.raw_data_buffer) == 0:
            if squared is not True:
                self.processed_data_buffer = np.zeros((im_data.shape[0], im_data.shape[1], 1))
            else:
                self.processed_data_buffer = np.zeros((im_data.shape[0], 1))
            self.raw_data_buffer = im_data

        elif self.raw_data_buffer.shape[2] < self.processing_window:
            self.raw_data_buffer = np.append(self.raw_data_buffer, im_data, axis=2)
            if squared is not True:
                self.processed_data_buffer = np.zeros((im_data.shape[0], im_data.shape[1], self.raw_data_buffer.shape[2]))
            else:
                self.processed_data_buffer = np.zeros((im_data.shape[0], self.raw_data_buffer.shape[2]))

        else:
            self.raw_data_buffer = np.append(self.raw_data_buffer[:, :, -self.processing_window + im_data.shape[2] :], im_data, axis=2)
            im_proc_tmp = self.raw_data_buffer
            average = np.mean(im_proc_tmp[:, :, -self.processing_window:], axis=2).reshape(-1, 3, 1)
            if squared:
                if accel:
                    average = abs(np.linalg.norm(average, axis=1) - 9.81)
                else:
                    average = np.linalg.norm(average, axis=1)

            if len(average.shape) == 3:
                if norm_min_bound or norm_max_bound:
                    for i in range(self.raw_data_buffer .shape[0]):
                        for j in range(self.raw_data_buffer .shape[1]):
                            if average[i, j, :] < 0:
                                average[i, j, :] = average[i, j, :] / abs(norm_min_bound)
                            elif average[i, j, :] >= 0:
                                average[i, j, :] = average[i, j, :] / norm_max_bound
                self.processed_data_buffer = np.append(self.processed_data_buffer[:, :, 1:], average, axis=2)

            else:
                if norm_min_bound or norm_max_bound:
                    for i in range(self.raw_data_buffer.shape[0]):
                        for j in range(self.raw_data_buffer.shape[1]):
                            if average[i, :] < 0:
                                average[i, :] = average[i, :] / abs(norm_min_bound)
                            elif average[i, :] >= 0:
                                average[i, :] = average[i, :] / norm_max_bound
                self.processed_data_buffer = np.append(self.processed_data_buffer[:, 1:], average, axis=1)

        return self.processed_data_buffer

    @staticmethod
    def get_peaks(
        new_sample: np.ndarray,
        signal: np.ndarray,
        signal_proc: np.ndarray,
        threshold: float,
        chanel_idx: Union[int, list] = None,
        nb_min_frame: float = 2000,
        is_one=None,
        min_peaks_interval=None,
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
                        sample_proc[i, j] = 1
                        is_one[i] = True
                    else:
                        sample_proc[i, j] = 0

        if len(signal) == 0:
            signal = new_sample
            signal_proc = sample_proc
            nb_peaks = None

        elif signal.shape[1] < nb_min_frame:
            signal = np.append(signal, new_sample, axis=1)
            signal_proc = np.append(signal_proc, sample_proc, axis=1)
            nb_peaks = None

        else:
            signal = np.append(signal[:, new_sample.shape[1] :], new_sample, axis=1)
            signal_proc = np.append(signal_proc[:, new_sample.shape[1] :], sample_proc, axis=1)

        if chanel_idx:
            signal = signal[chanel_idx, :]
            signal_proc = signal_proc[chanel_idx, :]

        if min_peaks_interval:
            signal_proc = RealTimeProcessing._check_and_adjust_intervall(signal_proc, min_peaks_interval)

        if isinstance(nb_peaks, list):
            nb_peaks.append(np.count_nonzero(signal_proc))
        return nb_peaks, signal_proc, signal, is_one

    @staticmethod
    def _check_and_adjust_intervall(signal, interval):
        for j in range(signal.shape[0]):
            if np.count_nonzero(signal[j, -interval:] == 1) not in [0, 1]:
                idx = np.where(signal[j, -interval:] == 1)[0]
                for i in idx[1:]:
                    signal[j, -interval:][i] = 0
        return signal

    @staticmethod
    def custom_processing(funct, data_tmp, **kwargs):
        return funct(data_tmp, **kwargs)


class OfflineProcessing(GenericProcessing):
    def __init__(self, data_rate: float = 2000, processing_window: int = 2000, moving_average_windows: int = 200):
        """
        Offline processing.
        """
        super(OfflineProcessing, self).__init__()
        self.data_rate = data_rate
        self.processing_window = processing_window
        self.moving_average_windows = moving_average_windows

    @staticmethod
    def custom_processing(funct, raw_data, **kwargs):
        return funct(raw_data, **kwargs)

    def process_emg(self, data: np.ndarray, mvc_list: list)-> np.ndarray:
        """
        Process EMG data.
        Parameters
        ----------
        data : np.ndarray
            Raw EMG data.
        mvc_list: list
            List of MVC for each muscle.

        Returns
        np.ndarray
            Processed EMG data.
        -------
    """
        return self._process_emg(data, mvc_list)

    @staticmethod
    def compute_mvc(
        nb_muscles: int,
        mvc_trials: np.ndarray,
        window_size: int,
        mvc_list_max: np.ndarray,
        tmp_file: str = None,
        output_file: str = None,
        save: bool = False,
    ):
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
        if tmp_file:
            os.remove(tmp_file)
        return mvc_list_max
