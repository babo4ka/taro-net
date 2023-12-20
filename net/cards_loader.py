import os

descs_path = "../cards_descs"

def load_descs(type):
    descs = []

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
                l = l.replace('(', '')
                l = l.replace(')', '')
                if type == 'general':
                    if i == 0 or i == 1:
                        descs += l.lower().split(' ')
                elif type == 'yn':
                    if i == 2:
                        descs += l.lower().split(' ')
                elif type == 'present':
                    if i == 3 or i == 5:
                        descs += l.lower().split(' ')
                elif type == 'past':
                    if i == 4 or i == 6:
                        descs += l.lower().split(' ')
                elif type == 'future':
                    if i == 7 or i == 8:
                        descs += l.lower().split(' ')
    return descs


