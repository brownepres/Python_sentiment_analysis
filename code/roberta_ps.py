import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import scipy.stats as stats
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def polarity_scores_roberta(txt):
    encoded_text = tokenizer(txt, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2],
        }
    return (scores_dict) 


def calc_roberta_compound(texts):
    roberta_compound_scores = []
    for text in texts:
        i_scores = polarity_scores_roberta(text[0:514])
        polarity_weights = torch.tensor([-1, 0, 1])
        probs = torch.tensor([i_scores['roberta_neg'],
                              i_scores['roberta_neu'],
                              i_scores['roberta_pos']])
        polarity = polarity_weights * probs
        polarity = polarity.sum(dim=-1)
        polarity_scaled = nn.Tanh()(polarity)
        polarity_score = polarity_scaled.item()
        roberta_compound_scores.append(polarity_score)
    return roberta_compound_scores