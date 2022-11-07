from biosiglive.streaming.connection import Server
import numpy as np


if __name__ == "__main__":
    server = Server(ip="localhost", port=50000, type="TCP")
    server.start()
    connection, message = server.client_listening()
    while True:
        data = {"emg_proc": np.ndarray((4, 200)), "emg_sample": 20}
        server.send_data(data, connection, message)
