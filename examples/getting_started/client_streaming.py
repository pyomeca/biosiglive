from biosiglive.streaming.client import Client, Message

message = Message(command=["emg"])
while True:
    client = Client("localhost", 50000)
    client.get_data(message)
