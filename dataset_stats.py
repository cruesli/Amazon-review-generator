
import numpy as np
import pandas as pd
import random
import re
from afinn import Afinn
from statistics import pvariance
from math import sqrt

#choosing the rating
rating = 1

#loading the dataset
data = pd.read_json('Sports_and_Outdoors_5.json', lines = True)
dff = data[data["overall"] == rating].copy()

reviewlist = dff["reviewText"].values.tolist()

#extracting the training dataset with 100 000 reviews per rating
text = reviewlist[:100000]   

#processing the reviews to only include words and gathering them in a list
def preprocess(text):
	text_input = re.sub('[^a-zA-Z1-9]+', ' ', str(text))
	output = re.sub(r'\d+', '',text_input)
	return output.lower().strip()

reviews = []

for i in text: 
    preprocess(i)
    reviews.append(preprocess(i))
    
#finding average length of reviews and no. of unique words
lengths=np.array([])
all_words = []

for r in reviews: 
    words = r.split()
    all_words +=words
    
    lengths=np.append(lengths, len(r.split()))

print(f"Unique words: {len(np.unique(np.array(all_words)))}")
print(f"Average length of reviews: {np.mean(lengths)}")

#random sample of 2000 from the training data    
reviews_1 = random.sample(reviews, 2000)

#using the afinn library to find the average sentiment value for the reviews and collecting in an array
afn = Afinn()
scores = np.array([])

for r in reviews_1: 
    scores = np.append(scores, afn.score(r))
    
print(scores)

#statistics of dataset-reviews
#mean of scores
me = np.mean(scores)
print(f"Mean score: {me}")

#population variance of scores
pvar = pvariance(scores)
print(f"Variance: {pvar}")

#calculation of sample size
n = 1.96**2 * (10/(0.15**2))
print(f"Sample size: {n}")

#confidence interval
confid_u = me + 1.96 * sqrt(pvar/2000)
confid_l = me - 1.96 * sqrt(pvar/2000)

print(f"Confidence interval : {[confid_l,confid_u]}")

