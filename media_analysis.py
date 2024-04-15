import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

plt.style.use("ggplot")

df = pd.read_csv('../sentiment_analysis_data/Reviews.csv')

sia = SentimentIntensityAnalyzer()

n = 100
k = 100

column_names = ["Sample" + str(i +1) for i in range(n)]
vader_text_sample = pd.DataFrame(columns=column_names)
vader_rating_sample= pd.DataFrame(columns=column_names)
vader_sample_column = pd.DataFrame(columns=["vader score"])
np.random.seed(2002)
for i in range(1):
    sampled_element = df.sample(n, replace=True)
    sampled_element_text = sampled_element['Text']
    
    for j in sampled_element_text:
        vader_score = sia.polarity_scores(j)
        vader_sample_column.loc[len(vader_sample_column)] = vader_score['compound']
        
    vader_text_sample.loc[i] = vader_sample_column["vader score"].values
    vader_sample_column = vader_sample_column.drop(vader_sample_column.index, axis=0)
    vader_rating_sample.loc[i] = sampled_element['Score'].values
    
vader_text_average = pd.DataFrame(np.mean(vader_text_sample.iloc[0:k, :], axis=0), columns=["vader_text_average"])
vader_rating_average = pd.DataFrame(np.mean(vader_rating_sample.iloc[0:k, :], axis=0), columns=["vader_rating_average"])

# Merge the DataFrames
vader_result_final = vader_text_average.merge(vader_rating_average, how="left", left_index=True, right_index=True)
   
melted_data = vader_result_final.head(20).reset_index().melt(id_vars='index')

# Create a figure and axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the text average as dots on the first y-axis
sns.scatterplot(data=melted_data[melted_data['variable'] == 'vader_text_average'],
                x='index', y='value', color='b', ax=ax1, label='Text Average')

# Create a second y-axis for ratings
ax2 = ax1.twinx()
# Plot the rating averages as dots on the second y-axis
sns.scatterplot(data=melted_data[melted_data['variable'] == 'vader_rating_average'],
                x='index', y='value', color='r', ax=ax2, label='Rating Average')

# Set y-axis limits according to the data ranges
ax1.set_ylim(0, 1)
ax2.set_ylim(1, 5)

# Set labels and titles
ax1.set_xlabel('Samples')
ax1.set_ylabel('Text Average', color='b')
ax2.set_ylabel('Rating Average', color='r')
plt.title('Text and Rating Averages')

# Show the legend
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Show the plot
plt.show()

vader_result_final['diff'] = vader_result_final['vader_rating_average'] - vader_result_final['vader_text_average']
std_of_vader_diff = np.std(vader_result_final['diff'])
correlation = vader_result_final['vader_text_average'].corr(vader_result_final['vader_rating_average'])

#---------------------------------------------------------------------------
#using roberta model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import torch.nn as nn

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

roberta_text_sample = pd.DataFrame(columns=column_names)
roberta_rating_sample= pd.DataFrame(columns=column_names)
roberta_sample_column = pd.DataFrame(columns=["roberta score"])
np.random.seed(2002)
for i in range(1):
    sampled_element = df.sample(n, replace=True)
    sampled_element_text = sampled_element['Text']
    
    for j in sampled_element_text:
        if len(j) > 512:
            # Slice the text to the first 512 characters
            j = j[:512]
        try:
            roberta_score = polarity_scores_roberta(j)
            
            roberta_sample_column.loc[len(roberta_sample_column)] = max(roberta_score.values())
        except RuntimeError:
            print(f"Broke for: {j}")
    roberta_text_sample.loc[i] = roberta_sample_column["roberta score"].values
    roberta_sample_column = roberta_sample_column.drop(roberta_sample_column.index, axis=0)
    roberta_rating_sample.loc[i] = sampled_element['Score'].values
    
roberta_text_average = pd.DataFrame(np.mean(roberta_text_sample.iloc[0:k, :], axis=0), columns=["roberta_text_average"])
roberta_rating_average = pd.DataFrame(np.mean(roberta_rating_sample.iloc[0:k, :], axis=0), columns=["roberta_rating_average"])

# Merge the DataFrames
roberta_result_final = roberta_text_average.merge(roberta_rating_average, how="left", left_index=True, right_index=True)



rtest = "I am extremely sad"

rtest_scores = polarity_scores_roberta(rtest)

# Convert scores to tensors
polarity_weights = torch.tensor([-1, 0, 1])
probs = torch.tensor([rtest_scores['roberta_neg'], rtest_scores['roberta_neu'], rtest_scores['roberta_pos']])
polarity = polarity_weights * probs
polarity = polarity.sum(dim=-1)
polarity_scaled = nn.Tanh()(polarity)
polarity_score = polarity_scaled.item()  # Extracting the scalar value from the tensor

print("RoBERTa Polarity Score:", polarity_score)
print(sia.polarity_scores(rtest))

print(rtest_scores['roberta_neg'] + rtest_scores['roberta_pos'])
print(sia.polarity_scores(rtest))











result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Text"]
    myid = row["Id"]
    result[myid] = sia.polarity_scores(text)
    
vader_result = pd.DataFrame(result).T
vader_result = vader_result.reset_index().rename(columns={'index':'Id'})
vader_result = vader_result.merge(df, how='left')

#check if vader results are aligned with the ratings
fig, axs = plt.subplots(1, 3, figsize=(15, 3))
sns.barplot(data=vader_result, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vader_result, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vader_result, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


#doing the same analysis with the pretrained model roberta, which takes the relationship
#between words into account


   
  
res = {}  
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row["Text"]
        myid = row["Id"]
        result_v = sia.polarity_scores(text)
        result_r = polarity_scores_roberta(text)
        both = {**result_v, **result_r}
        res[myid] = both
    except RuntimeError:
        print(f"Broke for {myid}")
                   

#venni több mintát(szöveges értékelés compoundja) a sokaságból és abból becsülni a sokasági compound
#értékére -> vader modell, vagy roberta modell csak meg kell nézni, hogy annak van egy compound-ja, 
#vagy a vader honnan kapja meg a saját compoundját

#probléma: nem normális eloszlású hanem exponenciális

#df eredetin végigmenni random 50 elemmel és megnézni azoknak az eloszlását
            



#vader modellel a compound értéket venni átlagnak -1 és 1 között, innen fogok becsülni
#a sokasági átlagra
#roberta modell esetében pedig a legnagyobb értéket venni átlagnak és onnan becsülni a 
#sokasági átlagra
