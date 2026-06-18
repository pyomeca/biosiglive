import numpy as np

from biosiglive import AsyncTCPServer
from biosiglive.streaming.utils import CircularBuffer
import asyncio
import threading
import time

if __name__ == "__main__":
    server = AsyncTCPServer("127.0.0.1", 12345)

    def task(data, t):
        pass

    threading.Thread(target=asyncio.run, args=(server.start(task=task),)).start()
    data_buffer = CircularBuffer(10, 1000)

    last_idx = None
    while True:
        time.sleep(0.05)
        if server.buffer is None:
            continue
        data, t = server.buffer.get_view()
        current_buff_idx = server.buffer.total_samples
        if last_idx:
            n_new_data = current_buff_idx - last_idx
            if n_new_data == 0:
                continue
            data = data[:, -n_new_data:]
            t = t[-n_new_data:]
        last_idx = current_buff_idx
        data_buffer.append(data, t)
        data_n, t_n = data_buffer.get_view()
