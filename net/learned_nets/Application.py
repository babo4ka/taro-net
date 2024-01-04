import datetime
import socket

from LearnedNet import LearnedNet

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

general_path = '../../../generated_texts/general/'
yn_path = '../../../generated_texts/yn/'
past_path = '../../../generated_texts/past/'
present_path = '../../../generated_texts/present/'
future_path = '../../../generated_texts/future/'

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
            fileName = (str(datetime.date.today()) + '_' + str(datetime.datetime.now().time().hour) + '_' + str(datetime.datetime.now().time().minute) + '_' + str(datetime.datetime.now().time().second)) + '.txt'
            file = open((general_path + fileName), 'w+')
            file.write(text)
            file.close()
        elif type == YN:
            text = yn_net.get_text(txt_len=25, temp=0.3)
            fileName = (str(datetime.date.today()) + '_' + str(datetime.datetime.now().time().hour) + '_' + str(
                datetime.datetime.now().time().minute) + '_' + str(datetime.datetime.now().time().second)) + '.txt'
            file = open((yn_path + fileName), 'w+')
            file.write(text)
            file.close()
        elif type == PAN:
            text = past_net.get_text(start_text='в гaдaнии')
            fileName = (str(datetime.date.today()) + '_' + str(datetime.datetime.now().time().hour) + '_' + str(
                datetime.datetime.now().time().minute) + '_' + str(datetime.datetime.now().time().second)) + '.txt'
            file = open((past_path + fileName), 'w+')
            file.write(text)
            file.close()
        elif type == PRN:
            text = present_net.get_text(txt_len=30, start_text='пpи гaдaнии')
            fileName = (str(datetime.date.today()) + '_' + str(datetime.datetime.now().time().hour) + '_' + str(
                datetime.datetime.now().time().minute) + '_' + str(datetime.datetime.now().time().second)) + '.txt'
            file = open((present_path + fileName), 'w+')
            file.write(text)
            file.close()
        elif type == FN:
            text = future_net.get_text(start_text='пpи гaдaнии')
            fileName = (str(datetime.date.today()) + '_' + str(datetime.datetime.now().time().hour) + '_' + str(
                datetime.datetime.now().time().minute) + '_' + str(datetime.datetime.now().time().second)) + '.txt'
            file = open((future_path + fileName), 'w+')
            file.write(text)
            file.close()

        conn.send(fileName.encode())

    conn.close()
