import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from net.taro_nets_utils import generate_text
from net.taro_nets_utils import get_batch
from net.taro_nets_utils import get_indexed
from net.taro_nets_utils import get_unique_words
from net.taro_nets_utils import load_descs
from net.taro_nets_utils import text_to_seq

descs = load_descs('yn', path='../../cards_descs')

uniqueWords = get_unique_words(descs)

indexes_to_words, words_to_indexes = get_indexed(uniqueWords)

sequence = text_to_seq(descs, words_to_indexes)

seq_len = 50
batch_size = 16

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

