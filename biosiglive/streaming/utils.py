import numpy as np


class CircularBuffer:
    def __init__(self, n, W, dtype=np.float64, time_dtype=np.float64, dt=None):
        self.W = W
        self.shape = (n, W)
        self.ring = np.zeros(self.shape, dtype=dtype)

        t = np.arange(W) * dt if dt is not None else np.arange(W)
        self.ring_t = None
        self.has_last = False
        self.total_samples = 0
        self.idx = 0
        self.full = False
        self.version = 0

        if dt is not None:
            self.dt = dt
            self.tol = dt * 1.5

    @property
    def empty(self):
        return self.idx == 0 and not self.full

    def append(
        self,
        x: np.ndarray,
        t: np.ndarray = None,
        fill_discontinuous: bool = False,
        fill_mode="single",
        row_idx=None,
        dt=None,
    ):
        """
        Parameters
        ----------
        x : np.ndarray
            The data to append. Shape: (n, w)
        t : np.ndarray, optional
            The timestamps corresponding to the data. Shape: (w,)
        fill_discontinuous : bool, optional
            Whether to fill discontinuities in the data with one NaN value. Mostly for use with pyqtgraph to plot gaps. Default is False.
        fill_mode : str, optional
            The mode for filling discontinuities. Default is "single". Possible values are "single" (fill with a single NaN value) and "full" (fill all missing data). Only used if fill_discontinuous is True.

        """
        row_idx = slice(None, None, None) if row_idx is None else row_idx

        self._append(x, t, row_idx)

    def _append(self, data, t=None, row_idx=None):
        """
        x shape: (n, w)
        """
        n = data.shape[1]
        start_version = self.version

        k = self.idx
        end = (k + n) % self.W

        if end <= k:
            split = self.W - k
            self.ring[:, k:] = data[:, :split]
            self.ring[:, :end] = data[:, split:]
            if t is not None:
                if self.ring_t is None:
                    self.ring_t = np.zeros(self.W, dtype=np.float64)
                self.ring_t[k:] = t[:split]
                self.ring_t[:end] = t[split:]

        else:
            self.ring[:, k:end] = data
            if t is not None:
                if self.ring_t is None:
                    self.ring_t = np.zeros(self.W, dtype=np.float64)
                self.ring_t[k:end] = t

        self.idx = end % self.W
        self.total_samples += n
        self.full |= self.total_samples >= self.W
        self.version = start_version + 1

    def get(self):
        while True:
            v1 = self.version

            # read all mutable state immediately after v1
            k = self.idx
            size = min(self.total_samples, self.W)

            start = (k - size) % self.W

            if start < k:
                data = self.ring[:, start:k].copy()
                t = self.ring_t[start:k].copy() if self.ring_t is not None else None
            else:
                data = np.concatenate((self.ring[:, start:], self.ring[:, :k]), axis=-1).copy()
                t = np.concatenate((self.ring_t[start:], self.ring_t[:k])).copy() if self.ring_t is not None else None

            v2 = self.version
            if v1 == v2:
                return data, t  # ← was silently dropping t

    def append_row(self, x, t=None, fill_discontinuous=False, fill_mode="single", row_idx=None):
        """Append a row to the buffer. The row_idx parameter specifies which row to append to. The rest of the rows will be filled with NaN values.
        Parameters
        ----------
        x : np.ndarray
            The data to append. Shape: (w,)
        t : np.ndarray, optional
            The timestamps corresponding to the data. Shape: (w,)
        fill_discontinuous : bool, optional
            Whether to fill discontinuities in the data with one NaN value. Mostly for use with pyqtgraph to plot gaps. Default is False.
        fill_mode : str, optional
            The mode for filling discontinuities. Default is "single". Possible values are "single" (fill with a single NaN value) and "full" (fill all missing data). Only used if fill_discontinuous is True.
        row_idx : int, optional
            The index of the row to append to. Default is 0.
        """
        row_idx = slice(None, None, None) if row_idx is None else row_idx

        self.append(x[np.newaxis, :], t, fill_discontinuous, fill_mode)


class RollingBuffer:
    def __init__(self, n, W):
        self.data = np.zeros((n, W), dtype=np.float32)
        self.time = np.zeros(W, dtype=np.float64)

    def append(self, x, t):
        w = x.shape[-1]

        self.data = np.roll(self.data, -w, axis=1)
        self.data[:, -w:] = x

        self.time = np.roll(self.time, -w)
        self.time[-w:] = t

    def get_time(self, len=None):
        return self.time

    def get_data(self, len=None):
        return self.data

    def get(self, len=None):
        return self.get_data(len), self.get_time(len)


def dic_merger(dic_to_merge: dict, new_dic: dict = None) -> dict:
    """Merge two dictionaries.

    Parameters
    ----------
    dic_to_merge : dict
        Existing dictionary to merge.
    new_dic : dict
        Temporary dictionary to merge with.

    Returns
    -------
    dict
        Merged dictionary.
    """

    if not new_dic:
        new_dic = dic_to_merge
    else:
        for key in dic_to_merge.keys():
            if dic_to_merge[key] is None:
                dic_to_merge[key] = [dic_to_merge[key]]
            if new_dic[key] is None:
                new_dic[key] = [new_dic[key]]
            if isinstance(new_dic[key], (int, float, str)):
                new_dic[key] = [new_dic[key]]
            if isinstance(dic_to_merge[key], (int, float, str)):
                dic_to_merge[key] = [dic_to_merge[key]]
            if isinstance(dic_to_merge[key], dict):
                if len(new_dic[key].keys()) == 0:
                    new_dic[key] = dic_to_merge[key]
                else:
                    new_dic[key] = dic_merger(dic_to_merge[key], new_dic[key])
            elif isinstance(dic_to_merge[key], list):
                new_dic[key] = dic_to_merge[key] + new_dic[key]
            elif isinstance(dic_to_merge[key], np.ndarray):
                if not isinstance(new_dic[key], np.ndarray):
                    new_dic[key] = np.array(new_dic[key])
                if len(new_dic[key].shape) == 1:
                    new_dic[key] = new_dic[key][:, np.newaxis]
                if len(dic_to_merge[key].shape) == 1:
                    dic_to_merge[key] = dic_to_merge[key][:, np.newaxis]
                new_dic[key] = np.append(dic_to_merge[key], new_dic[key], axis=-1)
            else:
                raise ValueError("Type not supported")
        for key in new_dic.keys():
            if key not in dic_to_merge.keys():
                new_dic[key] = new_dic[key]
    return new_dic
