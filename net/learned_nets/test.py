import socket

HOST = ('localhost', 9998)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(HOST)

client.send(b'1')
