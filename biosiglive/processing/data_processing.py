"""
This file contains the functions for data processing (offline and in real-time). Both class herites
 from the GenericProcessing class.
"""

from scipy.signal import butter, lfilter, filtfilt, convolve
import numpy as np
import os
import time
from typing import Union
from ..file_io.save_and_load import save, load


class GenericProcessing:
    def __init__(self):
        """
        Initialize the GenericProcessing class.
        """
        self.bpf_lcut = 10
        self.bpf_hcut = 425
        self.lpf_lcut = 5
        self.lp_butter_order = 4
        self.bp_butter_order = 2
        self.data_rate = None
        self.process_time = []

    @staticmethod
    def _butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5) -> tuple:
        """
        Create a butter bandpass filter.

        Parameters
        ----------
        lowcut : float
            Low cut frequency.
        highcut: float
            High cut frequency.
        fs: float
            Sampling frequency.
        order: int
            Order of the filter.

        Returns
        -------
        tuple
            Filter coefficients.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        return b, a

    @staticmethod
    def _butter_lowpass(lowcut, fs, order=4) -> tuple:
        """
        Create a butter lowpass filter.

        Parameters
        ----------
        lowcut : float
            Low cut frequency.
        fs: float
            Sampling frequency.
        order: int
            Order of the filter.

        Returns
        -------
        tuple
            Filter coefficients.
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        b, a = butter(order, [low], btype="low")
        return b, a

    def _butter_bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 5):
        """
        Apply a butter bandpass filter.

        Parameters
        ----------
        data: numpy.ndarray
            Data to filter.
        lowcut: float
            Low cut frequency.
        highcut: float
            High cut frequency.
        fs:     float
            Sampling frequency.
        order: int
            Order of the filter.

        Returns
        -------
        numpy.ndarray
            Filtered data.

        """
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_lowpass_filter(self, data: np.ndarray, lowcut: float, fs: float, order: int = 4) -> np.ndarray:
        """
        Apply a butter lowpass filter.

        Parameters
        ----------
        data: numpy.ndarray
            Data to filter.
        lowcut: float
            Low cut frequency.
        fs:    float
            Sampling frequency.
        order: int
            Order of the filter.

        Returns
        -------
        numpy.ndarray
            Filtered data.
        """
        b, a = self._butter_lowpass(lowcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def _moving_average(data: np.ndarray, window: np.ndarray, empty_ma: np.ndarray) -> np.ndarray:
        """
        Apply moving average.

        Parameters
        ----------
        data: numpy.ndarray
            Data to process.
        window: numpy.ndarray
            window to apply.
        empty_ma: numpy.ndarray
            Empty array to store the moving average.

        Returns
        -------
        numpy.ndarray
            Moving average.
        """
        for i in range(data.shape[0]):
            empty_ma[i, :] = convolve(data[i, :], window, mode="same", method="fft")
        return empty_ma

    @staticmethod
    def center(emg_data: np.ndarray, center_value: float = None) -> np.ndarray:
        """
        Center the EMG data.

        Parameters
        ----------
        emg_data : numpy.ndarray
            EMG data.
        center_value: int
            Value to center the data.

        Returns
        -------
        numpy.ndarray
            Centered EMG data.
        """
        center_value = center_value if center_value else emg_data.mean(axis=1)
        emg_centered = np.copy(emg_data)
        for i in range(emg_data.shape[0]):
            emg_centered[i, :] = emg_data[i, :] - center_value[i]
        return emg_centered

    @staticmethod
    def normalize_emg(emg_data: np.ndarray, mvc_list: list) -> np.ndarray:
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

    def calibration_matrix(self, data: np.ndarray, matrix: np.ndarray) -> np.ndarray:
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
        numpy.ndarray
            Calibrated data.
        """
        tic = time.time()
        data_cal = np.dot(matrix, data)
        self.process_time.append(time.time() - tic)
        return data_cal

    def process_generic_signal(
        self,
        data: np.ndarray,
        norm_values: Union[list, tuple] = None,
        band_pass_filter=True,
        low_pass_filter=False,
        moving_average=True,
        centering=True,
        absolute_value=True,
        normalization=False,
        moving_average_window=200,
        **kwargs,
    ) -> np.ndarray:
        """
        Process EMG data.

        Parameters
        ----------
        data : numpy.ndarray
            EMG data.
        norm_values : list
            Values to normalize the signal.
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
        moving_average_window : int
            Moving average window.

        Returns
        -------
        numpy.ndarray
            Processed EMG data.
        """
        self.update_signal_processing_parameters(**kwargs)
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
            data_proc = self.butter_lowpass_filter(data_proc, self.lpf_lcut, self.data_rate, order=self.lp_butter_order)
        elif moving_average:
            w = np.repeat(1, moving_average_window) / moving_average_window
            empty_ma = np.ndarray((data.shape[0], data.shape[1]))
            data_proc = self._moving_average(data_proc, w, empty_ma)

        if normalization:
            data_proc = self.normalize_emg(data_proc, norm_values)

        return data_proc

    def update_signal_processing_parameters(self, **kwargs):
        """
        update the signal processing parameters.

        Parameters
        ----------
        kwargs: dict
            Dictionary of parameters.
        """
        for key, value in kwargs.items():
            if key in self.__dict__:
                self.__dict__[key] = value


class RealTimeProcessing(GenericProcessing):
    def __init__(self, data_rate: Union[int, float], processing_window: int = None):
        """
        Initialize the class for real time processing.

        Parameters
        ----------
        data_rate : int
            Data rate.
        processing_window : int
            The window on which data will be processed.
        """
        super().__init__()
        self.data_rate = data_rate
        self.processing_window = processing_window if processing_window else data_rate
        self.raw_data_buffer = []
        self.processed_data_buffer = []
        self._is_one = None

    def process_emg(
        self,
        emg_data: np.ndarray,
        mvc_list: Union[list, tuple] = None,
        band_pass_filter=True,
        low_pass_filter=False,
        moving_average=True,
        centering=True,
        absolute_value=True,
        normalization=False,
        moving_average_window=200,
        **kwargs,
    ) -> np.ndarray:
        """
        Process EMG data in real-time.

        Parameters
        ----------
        emg_data : numpy.ndarray
            Temporary EMG data (nb_emg, emg_sample).
        mvc_list : list
            MVC values.
        band_pass_filter : bool
            True if apply band pass filter.
        low_pass_filter : bool
            True if apply low pass filter.
        moving_average : bool
            True if apply moving average.
        centering : bool
            True if apply centering.
        absolute_value : bool
            True if apply absolute value.
        normalization : bool
            True if apply normalization.
        moving_average_window : int
            Moving average window.

        Returns
        -------
        np.ndarray
           processed EMG data.

        """
        self.update_signal_processing_parameters(**kwargs)
        tic = time.time()
        if low_pass_filter and moving_average:
            raise RuntimeError("Please choose between low-pass filter and moving average.")
        ma_win = moving_average_window
        if ma_win > self.processing_window:
            raise RuntimeError(f"Moving average windows ({ma_win}) higher than emg windows ({self.processing_window}).")
        emg_sample = emg_data.shape[1]
        if emg_sample == 0:
            raise RuntimeError("EMG data are empty.")

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
            self.raw_data_buffer = emg_data
            processed_shape = self.raw_data_buffer.shape[1] if not moving_average else 1
            self.processed_data_buffer = np.zeros((emg_data.shape[0], processed_shape))

        elif self.raw_data_buffer.shape[1] < self.processing_window:
            self.raw_data_buffer = np.append(self.raw_data_buffer, emg_data, axis=1)
            if not moving_average:
                self.processed_data_buffer = np.append(
                    self.processed_data_buffer, np.zeros((emg_data.shape[0], emg_sample)), axis=1
                )
            else:
                self.processed_data_buffer = np.append(
                    self.processed_data_buffer, np.zeros((emg_data.shape[0], 1)), axis=1
                )

        else:
            self.raw_data_buffer = np.append(
                self.raw_data_buffer[:, -self.processing_window + emg_sample :], emg_data, axis=1
            )
            emg_proc_tmp = self.process_generic_signal(
                self.raw_data_buffer,
                band_pass_filter=band_pass_filter,
                centering=centering,
                absolute_value=absolute_value,
                low_pass_filter=False,
                moving_average=False,
                normalization=False,
            )
            self.processed_data_buffer = emg_proc_tmp / quot
            if low_pass_filter:
                self.processed_data_buffer = (
                    self.butter_lowpass_filter(emg_proc_tmp, self.lpf_lcut, self.data_rate, order=self.lp_butter_order)
                    / quot
                )
            elif moving_average:
                average = np.median(emg_proc_tmp[:, -ma_win:], axis=1).reshape(-1, 1)
                self.processed_data_buffer = np.append(self.processed_data_buffer[:, 1:], average / quot, axis=1)
        self.process_time.append(time.time() - tic)
        return self.processed_data_buffer

    def process_imu(
        self,
        im_data: np.ndarray,
        accel: bool = False,
        squared: bool = False,
        norm_min_bound: int = None,
        norm_max_bound: int = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Process IMU data in real-time.

        Parameters
        ----------
        im_data : numpy.ndarray
            Temporary IMU data (nb_imu, im_sample).
        accel : bool
            True if current data is acceleration data to adapt the processing.
        squared : bool
            True if apply squared.
        norm_min_bound : int
            Normalization minimum bound.
        norm_max_bound : int
            Normalization maximum bound.

        Returns
        -------
        np.ndarray
            processed IMU data.
        """
        self.update_signal_processing_parameters(**kwargs)
        tic = time.time()
        if len(self.raw_data_buffer) == 0:
            if squared is not True:
                self.processed_data_buffer = np.zeros((im_data.shape[0], im_data.shape[1], 1))
            else:
                self.processed_data_buffer = np.zeros((im_data.shape[0], 1))
            self.raw_data_buffer = im_data

        elif self.raw_data_buffer.shape[2] < self.processing_window:
            self.raw_data_buffer = np.append(self.raw_data_buffer, im_data, axis=2)
            if squared is not True:
                self.processed_data_buffer = np.zeros(
                    (im_data.shape[0], im_data.shape[1], self.raw_data_buffer.shape[2])
                )
            else:
                self.processed_data_buffer = np.zeros((im_data.shape[0], self.raw_data_buffer.shape[2]))

        else:
            self.raw_data_buffer = np.append(
                self.raw_data_buffer[:, :, -self.processing_window + im_data.shape[2] :], im_data, axis=2
            )
            im_proc_tmp = self.raw_data_buffer
            average = np.mean(im_proc_tmp[:, :, -self.processing_window :], axis=2).reshape(-1, 3, 1)
            if squared:
                if accel:
                    average = abs(np.linalg.norm(average, axis=1) - 9.81)
                else:
                    average = np.linalg.norm(average, axis=1)

            if len(average.shape) == 3:
                if norm_min_bound or norm_max_bound:
                    for i in range(self.raw_data_buffer.shape[0]):
                        for j in range(self.raw_data_buffer.shape[1]):
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
        self.process_time.append(time.time() - tic)
        return self.processed_data_buffer

    def get_peaks(
        self,
        new_sample: np.ndarray,
        threshold: float,
        min_peaks_interval=None,
    ) -> tuple:
        """
        Allow to get the number of peaks for an analog signal (to get cadence from treadmill for instance).

        Parameters
        ----------
        new_sample : numpy.ndarray
            New sample to add to the signal.
        threshold : float
            Threshold to detect peaks.
        min_peaks_interval : float
            Minimum interval between two peaks.

        Returns
        -------
        tuple
            Number of peaks and the processed signal.
        """
        tic = time.time()
        nb_peaks = []
        if len(new_sample.shape) == 1:
            new_sample = np.expand_dims(new_sample, 0)
        sample_proc = np.copy(new_sample)
        if not self._is_one:
            self._is_one = [False] * new_sample.shape[0]

        for i in range(new_sample.shape[0]):
            for j in range(new_sample.shape[1]):
                if new_sample[i, j] < threshold:
                    sample_proc[i, j] = 0
                    self._is_one[i] = False
                elif new_sample[i, j] >= threshold:
                    if not self._is_one[i]:
                        sample_proc[i, j] = 1
                        self._is_one[i] = True
                    else:
                        sample_proc[i, j] = 0

        if len(self.raw_data_buffer) == 0:
            self.raw_data_buffer = new_sample
            self.processed_data_buffer = sample_proc
            nb_peaks = None

        elif self.raw_data_buffer.shape[1] < self.processing_window:
            self.raw_data_buffer = np.append(self.raw_data_buffer, new_sample, axis=1)
            self.processed_data_buffer = np.append(self.processed_data_buffer, sample_proc, axis=1)
            nb_peaks = None

        else:
            self.raw_data_buffer = np.append(self.raw_data_buffer[:, new_sample.shape[1] :], new_sample, axis=1)
            self.processed_data_buffer = np.append(
                self.processed_data_buffer[:, new_sample.shape[1] :], sample_proc, axis=1
            )

        if min_peaks_interval:
            self.processed_data_buffer = RealTimeProcessing._check_and_adjust_interval(
                self.processed_data_buffer, min_peaks_interval
            )

        if isinstance(nb_peaks, list):
            nb_peaks.append(np.count_nonzero(self.processed_data_buffer))
        self.process_time.append(time.time() - tic)
        return nb_peaks, self.processed_data_buffer

    @staticmethod
    def _check_and_adjust_interval(signal, interval):
        for j in range(signal.shape[0]):
            if np.count_nonzero(signal[j, -interval:] == 1) not in [0, 1]:
                idx = np.where(signal[j, -interval:] == 1)[0]
                for i in idx[1:]:
                    signal[j, -interval:][i] = 0
        return signal

    def custom_processing(self, funct: callable, data_tmp: np.ndarray, **kwargs) -> np.ndarray:
        """
        Allow to apply a custom processing function to the data.

        Parameters
        ----------
        funct: callable
            Function to apply to the data.
        data_tmp: numpy.ndarray
            Data to process.

        Returns
        -------
        numpy.ndarray
            Processed data.
        """
        tic = time.time()
        data_tmp = funct(data_tmp, **kwargs)
        self.process_time.append(time.time() - tic)
        return data_tmp

    def get_mean_process_time(self):
        return np.mean(self.process_time)


class OfflineProcessing(GenericProcessing):
    def __init__(self, data_rate: float = None, processing_window: int = None):
        """
        Offline processing.

        Parameters
        ----------
        data_rate : float
            Data rate of the signal.
        processing_window : int
            Number of samples to process at once.
        """
        super(OfflineProcessing, self).__init__()
        self.data_rate = data_rate
        self.processing_window = processing_window

    def process_emg(self, data: np.ndarray, mvc_list: list = None, **kwargs) -> np.ndarray:
        """
        Process EMG data.

        Parameters
        ----------
        data : np.ndarray
            Raw EMG data.
        mvc_list: list
            List of MVC for each muscle.

        Returns
        -------
        np.ndarray
            Processed EMG data.

        """
        return self.process_generic_signal(data, mvc_list, **kwargs)

    @staticmethod
    def compute_mvc(
        nb_muscles: int,
        mvc_trials: np.ndarray,
        window_size: int,
        tmp_file: str = None,
        output_file: str = None,
        save_file: bool = False,
    ) -> list:
        """
        Compute MVC from several mvc_trials.

        Parameters
        ----------
        nb_muscles : int
            Number of muscles.
        mvc_trials : numpy.ndarray
            EMG data for all trials.
        window_size : int
            Size of the window to compute MVC. Usually it is 1 second so the data rate.
        tmp_file : str
            Name of the temporary file.
        output_file : str
            Name of the output file.
        save_file : bool
            If true, save the results.

        Returns
        -------
        list
            MVC for each muscle.

        """
        mvc_list_max = []
        for i in range(nb_muscles):
            mvc_temp = -np.sort(-mvc_trials, axis=1)
            if i == 0:
                mvc_list_max = mvc_temp[:, :window_size]
            else:
                mvc_list_max = np.concatenate((mvc_list_max, mvc_temp[:, :window_size]), axis=1)
        mvc_list_max = -np.sort(-mvc_list_max, axis=1)[:, :window_size]
        mvc_list_max = np.median(mvc_list_max, axis=1)

        if tmp_file:
            mat_content = load(tmp_file)
            mat_content["MVC_list_max"] = mvc_list_max
        else:
            mat_content = {"MVC_list_max": mvc_list_max, "MVC_trials": mvc_trials}

        if save_file:
            save(mat_content, output_file)
        if tmp_file:
            os.remove(tmp_file)
        return mvc_list_max
