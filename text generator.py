import argparse
import torch
import numpy as np
from model import Model
from dataset import Dataset
import random
from afinn import Afinn
from statistics import variance, pvariance

device = torch.device ("cuda")

parser = argparse.ArgumentParser()
parser.add_argument("--rt_num", type=int, default = 1)
parser.add_argument('--max-epochs', type=int, default = 10)
parser.add_argument('--batch-size', type=int, default = 128)
parser.add_argument('--sequence-length', type=int, default = 4)
args = parser.parse_args()



dataset = Dataset(args)
model = Model(dataset)
model.to(device)

# loads a state from a (hopefully) trained model
model.load_state_dict(torch.load(f"States/Model_{args.rt_num}_star2.0", map_location="cuda"))

# Generates the text by predicting the next words.
def predict(dataset, model, text, next_words = 30):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))
    state_h = state_h.to(device)
    state_c = state_c.to(device)

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]], device = device)
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits.cpu(), dim=0).detach().numpy()
        word_index = np.random.choice(len(last_word_logits), p=p)
        words.append(dataset.index_to_word[word_index])
    return words

# Word to initialize the predictor, we choose 6 different. We tried to choose neutral seeds to keep bias minimal
seed = ["i", "my", "this", "in", "the", "it"] 

# using the afinn function to rate the reviews on sentiment
afn = Afinn()
scores = np.array([])

# example text
sn =" ".join(predict(dataset, model, text = "i")).capitalize()+"."
print(sn)
print(afn.score(sn))

# Generates 2000 reviews, scores them and appends the score to a list
for i in range(2000):
	
	s = " ".join(predict(dataset, model, text= random.choice(seed))).capitalize()+"."
	#print(s)
	#print(afn.score(s))
	scores = np.append(scores, afn.score(s))

# Mean of scores
mean = np.mean(scores)
print(f"Mean: {mean}")

# population variance of scores
pvar = pvariance(scores)
print(f"Variance: {pvar}")

# confidence interval
confid_u = mean + 1.96 * np.sqrt(pvar/2000)
confid_l = mean - 1.96 * np.sqrt(pvar/2000) 

print(f"Confidence interval: {[confid_l,confid_u]}")
	