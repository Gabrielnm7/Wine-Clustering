# I'm just testing the dockerization, dismiss this.

import time 
import socket
import pandas as pd

url = "https://storage.googleapis.com/the_public_bucket/wine-clustering.csv"

data_wine = pd.read_csv(url)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0",9999))

server.listen()

while True:
    client, addr = server.accept()
    print("Connection from: ", addr)
    client.send("You are connected! \n".encode())
    client.send(f"{data_wine.head()}\n".encode())
    time.sleep(2)
    client.send("You are being disconnected \n".encode())
    client.close()