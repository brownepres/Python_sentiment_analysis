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

plt.style.use("ggplot")

df = pd.read_csv('../sentiment_analysis_data/Reviews.csv')

sia = SentimentIntensityAnalyzer()

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


n=1000
sample = pd.DataFrame(columns=['Text', 'VADER_compound', 'roBERTa_compound', 'score'])
sample['Text'] = df.Text.sample(n, replace=True, ignore_index=True, random_state=2002)

#vader compounds to the sample
vader_compound_scores = []
for i in sample.Text:
    vader_compound_scores.append(sia.polarity_scores(i)['compound'])
    
sample['VADER_compound'] = vader_compound_scores

#scores to the sample
sample['score'] = df.Score.sample(n, replace=True, ignore_index=True, random_state=2002)

#roberta compounds to the sample
roberta_compound_scores = []
for i in sample.Text:
    i_scores = polarity_scores_roberta(i[0:514])
    polarity_weights = torch.tensor([-1, 0, 1])
    probs = torch.tensor([i_scores['roberta_neg'],
                          i_scores['roberta_neu'],
                          i_scores['roberta_pos']])
    polarity = polarity_weights * probs
    polarity = polarity.sum(dim=-1)
    polarity_scaled = nn.Tanh()(polarity)
    polarity_score = polarity_scaled.item()
    roberta_compound_scores.append(polarity_score)

sample['roBERTa_compound'] = roberta_compound_scores

vader_corr = sample['score'].corr(sample['VADER_compound'])
roberta_corr = sample['score'].corr(sample['roBERTa_compound'])
print(vader_corr, roberta_corr)

vader_mean = np.mean(sample['VADER_compound'])
roberta_mean = np.mean(sample['roBERTa_compound'])
score_mean = np.mean(sample['score'])
print(vader_mean, roberta_mean, score_mean)

#95% percent certainity for the entire dataset
alpha = 1-0.95
z = stats.norm.ppf(1-(alpha/2))
SE_vader = stats.sem(sample['VADER_compound'])
delta_vader = z*SE_vader
SE_roberta = stats.sem(sample['roBERTa_compound'])
delta_roberta = z*SE_roberta
SE_score = stats.sem(sample['score'])
delta_score = z*SE_score
print([np.mean(sample['VADER_compound'])-delta_vader, np.mean(sample['VADER_compound'])+delta_vader])
print([np.mean(sample['roBERTa_compound'])-delta_roberta, np.mean(sample['roBERTa_compound'])+delta_roberta])
print([np.mean(sample['score'])-delta_score, np.mean(sample['score'])+delta_score])

#-----------------------------------------------------------------------------
#Misleading VADER model
#Take the 30 data from the sample where the VADER and roBERTa compounds are the furthest from each other
sample['diff'] = abs(sample['VADER_compound'] - sample['roBERTa_compound'])
sample_sorted_diff = sample.sort_values(by='diff', ascending=False, ignore_index=True).head(30)
sorted_corr_vader = sample_sorted_diff['score'].corr(sample_sorted_diff['VADER_compound'])
sorted_corr_roberta = sample_sorted_diff['score'].corr(sample_sorted_diff['roBERTa_compound'])

#in critical cases like this 30, the roBERTa model maintains a strong correlation with 'score'
#while the VADER model fails to interpret the majority of these texts
#it is due to the fact that VADER model only takes words, while roBERTa takes context as well
#into account when calculating sentiment value
print(sorted_corr_roberta, sorted_corr_vader)


fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.scatter(sample_sorted_diff.index, sample_sorted_diff['VADER_compound'], label='VADER_compound', color='blue', marker='o', edgecolor='black')
ax1.scatter(sample_sorted_diff.index, sample_sorted_diff['roBERTa_compound'], label='roBERTa_compound', color='red', marker='x', edgecolor='black')
ax1.set_ylabel('VADER and roBERTa compound scores')
ax1.set_ylim(-1, 1.5)
ax2 = ax1.twinx()
ax2.scatter(sample_sorted_diff.index, sample_sorted_diff['score'], label='score', color='green', marker='s', edgecolor='black')
ax2.set_ylabel('Score')
ax2.set_ylim(0, 6)
plt.xlabel('Index')
plt.title('The top 30 largest difference between VADER roBERTa model compounds, compared to score')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)
ax2.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

    












