from LearnedNet import LearnedNet
import socket

from net.learning_nets.GeneralMeaningNet import TaroGenNet
from net.learning_nets.PastNet import PastNet
from net.learning_nets.FutureNet import FutureNet
from net.learning_nets.PresentNet import PresentNet
from net.learning_nets.YesNoNet import YNNet

general_net = LearnedNet('./general_meaning/GeneralMeaningNet_temp_3.pt', 'general')
yn_net = LearnedNet('./yes_no/YNNet_temp_3.pt', 'yn')
past_net = LearnedNet('./past/PastNet_temp_4.pt', 'past')
present_net = LearnedNet('./present/PresentNet_temp_3.pt', 'present')
future_net = LearnedNet('./future/FutureNet_temp_2.pt', 'future')


GN = 1
YN = 2
PAN = 3
PRN = 4
FN = 5

HOST = ('localhost', 9998)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(HOST)
server.listen()
print('work')

while True:
    conn, addr = server.accept()
    print(addr)

    data = conn.recv(2048)

    if data:
        type = int(data.decode())
        if type == GN:
            text = general_net.get_text(txt_len=25, temp=0.4)
        elif type == YN:
            text = yn_net.get_text(txt_len=25, temp=0.3)
        elif type == PAN:
            text = past_net.get_text()
        elif type == PRN:
            text = present_net.get_text(txt_len=30)
        elif type == FN:
            future_net.get_text()

        print(text)

    conn.close()
