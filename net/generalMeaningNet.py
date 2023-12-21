import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

from cards_loader import load_descs

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def generate_text(net, start_text='карта указывает', pred_len=200, temp=0.3):
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


class TaroGenNet(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TaroGenNet, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


net = TaroGenNet(input_size=len(indexes_to_words), hidden_size=128, embedding_size=128, n_layers=2)
net.to(device)

loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01, amsgrad=True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    verbose=True,
    factor=0.5
)

epochs = 5000
loss_his = []
loss_history = []

for epoch in range(epochs):
    net.train()
    train, target = get_batch(sequence)
    train = train.permute(1, 0, 2).to(device)
    target = target.permute(1, 0, 2).to(device)
    hidden = net.init_hidden(batch_size)

    output, hidden = net(train, hidden)
    loss = loss_fun(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    loss_history.append(loss.detach().cpu())

    loss_his.append(loss.item())
    if(len(loss_his) >= 50):
        mean_loss = np.mean(loss_his)
        print('loss: ', mean_loss)
        scheduler.step(mean_loss)
        loss_his = []
        net.eval()
        predicted_text = generate_text(net)
        print(predicted_text)


plt.plot(loss_history)
plt.show()