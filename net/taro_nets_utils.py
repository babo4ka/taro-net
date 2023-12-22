import os
import torch
import numpy as np
import torch.nn.functional as F

def load_descs(type, path="../cards_descs"):
    descs = []

    for filename in os.listdir(path):
        with open((path + "/" + filename), encoding='utf-8') as f:
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
                        descs.append(l.lower().split(' '))
                elif type == 'yn':
                    if i == 2:
                        descs.append(l.lower().split(' '))
                elif type == 'present':
                    if i == 3 or i == 5:
                        descs.append(l.lower().split(' '))
                elif type == 'past':
                    if i == 4 or i == 6:
                        descs.append(l.lower().split(' '))
                elif type == 'future':
                    if i == 7 or i == 8:
                        descs.append(l.lower().split(' '))
    return descs


def get_unique_words(array):
    unique_words = []

    for arr in array:
        for word in arr:
            if not word in unique_words:
                unique_words.append(word)

    return unique_words


def get_indexed(words_array):
    indexes_to_words = {}
    words_to_indexes = {}

    for i, word in enumerate(words_array):
        indexes_to_words[i] = word
        words_to_indexes[word] = i

    return indexes_to_words, words_to_indexes


def text_to_seq(text, words_to_indexes):
    seq = []

    for arr in text:
        temp = []
        for word in arr:
            temp.append(words_to_indexes[word])

        seq.append(temp)

    return seq


def get_batch(seq, batch_size, seq_len):
    trains = []
    targets = []

    for _ in range(batch_size):
        batch_start = np.random.randint(0, len(seq) - seq_len)
        chunk = seq[batch_start: batch_start + seq_len]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)


def generate_text(net, words_to_indexes, indexes_to_words, device, start_text='карта указывает', pred_len=50, temp=0.3):
    hidden = net.init_hidden()
    idx_inp = [words_to_indexes[str] for str in start_text.split(' ')]
    train = torch.LongTensor(idx_inp).view(-1, 1, 1).to(device)
    pred_text = start_text

    _, hidden = net(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(pred_len):
        output, hidden = net(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(words_to_indexes), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        pred_word = indexes_to_words[top_index]
        pred_text = pred_text + " " + pred_word

    return pred_text