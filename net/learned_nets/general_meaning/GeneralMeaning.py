import torch

from net.taro_nets_utils import generate_text, load_descs, get_unique_words, get_indexed


class GeneralMeaning:
    def __init__(self):
        self.net = torch.load('/GeneralMeaningNet_temp_3.pt')
        self.net.eval()

        self.descs = load_descs('general', path='../../cards_descs')

        self.uniqueWords = get_unique_words(self.descs)

        self.indexes_to_words, self.words_to_indexes = get_indexed(self.uniqueWords)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net.to(self.device)

    def get_text(self, start_text='эта карта', txt_len=25, temp=0.3):
        return generate_text(self.net, self.words_to_indexes, self.indexes_to_words, self.device, start_text=start_text,
                             pred_len=txt_len, temp=temp)
