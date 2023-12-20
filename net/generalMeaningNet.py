from cards_loader import load_descs

import torch
import torch.nn as nn
import numpy as np

descs = load_descs('general')

uniqueWords = []

for word in descs:
    if not word in uniqueWords:
        uniqueWords.append(word)

indexes_to_words = {}
words_to_indexes = {}

for i, word in enumerate(uniqueWords):
    indexes_to_words[i] = word
    words_to_indexes[word] = i


def text_to_seq():
    seq = []

    for desc in descs:
        seq.append(words_to_indexes[desc])

    return seq


sequence = text_to_seq()

seq_len = 256
batch_size = 16


def get_batch(seq):
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


def generate_test(net, start_text='карта', pred_len=200, temp=0.3):
    hidden = net.init_hidden()
    idx_inp = [words_to_indexes[str] for str in start_text]
