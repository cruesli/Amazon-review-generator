import torch
import pandas as pd
from collections import Counter
import re


device = torch.device("cuda")

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
    ):
        self.args = args
        self.words = self.load_words()
        self.uniq_words = self.get_uniq_words()


        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    # takes inn the dataset and filters out the reviewtexts, removes icons and numbers and then adds all the words into a list
    def load_words(self):
		
        train_df = pd.read_json('Sports_and_Outdoors_5.json', lines = True)
        dff = train_df[train_df["overall"] == self.args.rt_num].copy()
        text = dff['reviewText'].values.tolist()
        
        # dividing by a number to get roughly 100 000 reviews per rating
        div = [1,1,2,5,19]
        
        text = text[:int(len(text)/div[self.args.rt_num-1])]
        
        output = re.sub('[^a-zA-Z]+', ' ', str(text))
        return output.lower().split(" ")
    # counts the occurences of each word and returns them in a sorted list from most frequent to least
    def get_uniq_words(self):
        word_counts = Counter(self.words)

        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args.sequence_length
    # Turns the words into tensors
    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args.sequence_length], device = device),
            torch.tensor(self.words_indexes[index+1:index+self.args.sequence_length+1], device = device),
        )
