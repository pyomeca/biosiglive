import asyncio
import struct
import numpy as np


class AsyncTCPClient:
    """
    Async TCP client for sending NumPy arrays.

    Parameters
    ------------
    host : str
    port : int

    Returns
    --------
    None
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.writer = None

    async def connect(self) -> None:
        """
        Connect to server.

        Parameters
        ------------
        None

        Returns
        --------
        None
        """
        try:
            _, self.writer = await asyncio.wait_for(asyncio.open_connection(self.host, self.port), timeout=self.timeout)
        except asyncio.TimeoutError:
            print(f"Connection to {self.host}:{self.port} timed out after {self.timeout} seconds.")

    async def send_array(self, array: np.ndarray, sample_time: np.ndarray = None) -> None:
        """
        Send NumPy array.

        Parameters
        ------------
        array : np.ndarray
        sample_time : np.ndarray, optional
            The timestamps corresponding to each sample in the array. If provided, the timestamps will be sent along with the array. The server should be able to handle this case.
        Returns
        --------
        None
        """
        if self.writer is None:
            raise RuntimeError("Client not connected")
        if array.dtype != np.float64:
            array = array.astype(np.float64)

        if sample_time is not None:
            if sample_time.dtype != np.float64:
                sample_time = sample_time.astype(np.float64)
            if len(sample_time) != array.shape[-1]:
                return
            array = np.vstack((sample_time, array))

        payload = array.tobytes()

        shape = array.shape
        ndim = len(shape)

        is_time = struct.pack("!?", sample_time is not None)
        header = struct.pack("!II", len(payload), ndim)
        shape_bytes = struct.pack(f"!{ndim}I", *shape)

        self.writer.write(is_time + header + shape_bytes + payload)
        await self.writer.drain()

    async def close(self) -> None:
        """
        Close connection.

        Parameters
        ------------
        None

        Returns
        --------
        None
        """
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
