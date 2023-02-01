# TCP/IP server

This example shows how to start a server to send data over a TCP/IP connection.
The server is first created with the IP address and port as input, and then it is started. In the loop, the server waits for a connection from a client. If a client sends a message to the server, the server reads the message and sends data based on the connection that was just established.
To try this example, please run this code, then the get_from_server.py example.

```
from biosiglive import Server
import numpy as np

if __name__ == '__main__':
    server_ip = "127.0.0.1"
    port = 50000
    server = Server(server_ip, port)
    server.start()
    while True:
        print("Server is listening")
        connection, message = server.client_listening()
        data = {"proc_device_data": np.random.rand(5, 100), "marker_set_data": np.random.rand(15, 100)}
        server.send_data(data, connection, message)
        print("Data sent to the client.")
```
