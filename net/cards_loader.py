import os
import re

descs_path = "../cards_descs"

general_descs = []
yn_descs = []
past_descs = []
present_descs = []
future_descs = []


def load_descs():
    for filename in os.listdir(descs_path):
        with open(("../cards_descs/" + filename), encoding='utf-8') as f:
            lines = [line.rstrip() for line in f]
            lines.pop(0)
            for l in lines:
                if l == "===":
                    lines.remove(l)

            for i, l in enumerate(lines):
                l = l.replace(',', '')
                l = l.replace('.', '')
                l = l.replace('!', '')
                l = l.replace('?', '')
                if i == 0 or i == 1:
                    general_descs.append(l.lower().split(' '))
                elif i == 2:
                    yn_descs.append(l.lower().split(' '))
                elif i == 3 or i == 5:
                    present_descs.append(l.lower().split(' '))
                elif i == 4 or i == 6:
                    past_descs.append(l.lower().split(' '))
                elif i == 7 or i == 8:
                    future_descs.append(l.lower().split(' '))


load_descs()
