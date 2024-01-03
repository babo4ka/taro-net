import os

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
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True))


if __name__ == '__main__':
    descs = load_descs('general', path='../../cards_descs')

    uniqueWords = get_unique_words(descs)

    indexes_to_words, words_to_indexes = get_indexed(uniqueWords)

    sequence = text_to_seq(descs, words_to_indexes)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net = TaroGenNet(input_size=len(indexes_to_words), hidden_size=128, embedding_size=128, n_layers=2)
    net.to(device)

    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        verbose=True,
        factor=0.5
    )

    epochs = 2500
    loss_his = []
    loss_history = []

    gen_loss_history = []
    gen_loss_epochs = []

    temp_num = 1

    batch_size = 16

    for epoch in range(epochs):
        ep_loss = []
        for seq in sequence:
            net.train()
            train, target = get_batch(seq, batch_size)
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

        if len(loss_his) >= 50:
            mean_loss = np.mean(loss_his)
            print('loss: ', mean_loss)
            print('epoch:', epoch)
            scheduler.step(mean_loss)
            loss_his = []
            net.eval()

        loss_history.append(np.mean(ep_loss))

        if epoch % 100 == 0 and epoch != 0:
            print('epoch:', epoch)

            gen_loss_history.append(np.mean(ep_loss))
            gen_loss_epochs.append(epoch)

            plt.plot(loss_history, c='black', label='потери сети общее')
            plt.plot(gen_loss_epochs, gen_loss_history, marker='.', c='pink', label='потери на поколении')
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
                minimal_loss = np.min(gen_loss_history)
                index = gen_loss_history.index(minimal_loss)
                best_epoch = gen_loss_epochs[index]
                nums_to_delete = [int(num / 100) for num in gen_loss_epochs if num != best_epoch]

                print('saving gen ', best_epoch, ' with loss = ', minimal_loss)

                for i in nums_to_delete:
                    os.remove("../learned_nets/general_meaning/GeneralMeaningNet_temp_" + str(i) + ".pt")
                break
