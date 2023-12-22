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

descs = load_descs('general', path='../../cards_descs')

uniqueWords = get_unique_words(descs)

indexes_to_words, words_to_indexes = get_indexed(uniqueWords)

sequence = text_to_seq(descs, words_to_indexes)

seq_len = 50
batch_size = 16

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

epochs = 2500
loss_his = []
loss_history = []

temp_num = 1

for epoch in range(epochs):
    ep_loss = []
    for seq in sequence:
        net.train()
        train, target = get_batch(seq, batch_size, seq_len)
        train = train.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        hidden = net.init_hidden(batch_size)

        output, hidden = net(train, hidden)
        loss = loss_fun(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        ep_loss.append(loss.detach().cpu())

        loss_his.append(loss.item())
        if (len(loss_his) >= 50):
            mean_loss = np.mean(loss_his)
            print('loss: ', mean_loss)
            print('epoch:', epoch)
            scheduler.step(mean_loss)
            loss_his = []
            net.eval()

    loss_history.append(np.mean(ep_loss))

    if epoch % 100 == 0:
        print('epoch:', epoch)

        plt.plot(loss_history, c='pink', label='потери сети общее')
        plt.xlabel('эпохи')
        plt.ylabel('потери')
        plt.legend(loc='upper left')
        plt.show()

        torch.save(net, ("../learned_nets/general_meaning/GeneralMeaningNet_temp_" + str(temp_num) + ".pt"))
        temp_num += 1

        predicted_text = generate_text(net, words_to_indexes, indexes_to_words, device, start_text='эта карта')
        print(predicted_text)

        print('1 - прекратить, любой символ или слово - продолжить')
        action = input()
        if action == '1':
            break

torch.save(net, "../learned_nets/general_meaning/GeneralMeaningNet.pt")
