import asyncio
import struct
import numpy as np
import time
from .utils import CircularBuffer, RollingBuffer


class AsyncTCPServer:
    """
    Async TCP server for receiving NumPy arrays.

    Parameters
    ------------
    host : str
    port : int

    Returns
    --------
    None
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000, buffer_length: int = 1000):
        self.host = host
        self.port = port
        self.server = None
        self.buffer = None
        self.buffer_length = buffer_length
        self.dt = 0.01

    def init_buffer(self, n_channels, dt=None):
        if dt is not None:
            self.dt = dt
        self.buffer = CircularBuffer(n_channels, self.buffer_length, dtype=np.float64, dt=self.dt)
        # self.buffer = RollingBuffer(n_channels, self.buffer_length)

    async def start(self, task: callable = None) -> None:
        """
        Start the TCP server.

        Parameters
        ------------
        task : callable
            Optional task to run when the server receive new data. The task should be a function that takes a NumPy array as input. Becarefull, the task will be called for each new data received, so it should be fast to avoid blocking the server.

        Returns
        --------
        None
        """
        self.task = task
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        # print(f"Server started on {self.host}:{self.port}")

        async with self.server:
            await self.server.serve_forever()

    async def handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """
        Handle incoming client connection.

        Parameters
        ------------
        reader : asyncio.StreamReader
        writer : asyncio.StreamWriter

        Returns
        --------
        None
        """
        addr = writer.get_extra_info("peername")
        try:
            while True:
                data, t = await self._read_array(reader)
                if data is None:
                    break
                if self.buffer is None:
                    self.init_buffer(data.shape[0])

                # if data.shape[0] != self.buffer.shape[0]:
                #     raise ValueError(f"Received data with shape {data.shape}, but expected {self.buffer.shape}.")
                # if np.any(np.diff(t) <= 0):
                #     print(" intra-chunk non-monotonic")
                self.buffer.append(data, t)
                self.task(data, t) if self.task else None

        except asyncio.IncompleteReadError:
            pass

        finally:
            # print(f"Client disconnected: {addr}")
            writer.close()
            # await writer.wait_closed()

    async def _read_array(self, reader: asyncio.StreamReader) -> tuple:
        """
        Read a NumPy array from stream.

        Parameters
        ------------
        reader : asyncio.StreamReader

        Returns
        --------
        tuple
        """
        is_time = await reader.readexactly(1)
        is_time = struct.unpack("!?", is_time)[0]

        header = await reader.readexactly(8)
        size, ndim = struct.unpack("!II", header)

        shape_bytes = await reader.readexactly(4 * ndim)
        shape = struct.unpack(f"!{ndim}I", shape_bytes)

        payload = await reader.readexactly(size)
        array = np.frombuffer(payload, dtype=np.float64).reshape(shape)

        if is_time:
            t = array[0, :]
            array = array[1:, :]
            if self.dt is None:
                self.dt = t[1] - t[0]
                self.tol = self.dt * 1.5
        else:
            t = None

        return array, t

    async def stop(self) -> None:
        """
        Stop the TCP server.

        Parameters
        ------------
        None

        Returns
        --------
        None
        """
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            # print("Server stopped")

    async def get_data(self) -> tuple:
        """
        Get the current data and timestamps in the buffer.

        Parameters
        ------------
        None

        Returns
        --------
        tuple
        """
        if self.buffer is not None:
            return self.buffer.get()
        else:
            raise ValueError("Buffer is empty. No data received yet.")
