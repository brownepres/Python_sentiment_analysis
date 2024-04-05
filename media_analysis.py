import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

plt.style.use("ggplot")

df = pd.read_csv('../sentiment_analysis_data/Reviews.csv')
df = df.head(500)

sia = SentimentIntensityAnalyzer()

result = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["Text"]
    myid = row["Id"]
    result[myid] = sia.polarity_scores(text)
    
vader_result = pd.DataFrame(result).T
vader_result = vader_result.reset_index().rename(columns={'index':'Id'})
vader_result = vader_result.merge(df, how='left')

#check if vader results are aligned with the ratings
