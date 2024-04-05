import pandas as pd
import numpy as np
import seaborn as sns
import nltk
import matplotlib.pyplot as plt
plt.style.use("ggplot")

df = pd.read_csv('../sentiment_analysis_data/Reviews.csv')
df = df.head(500)
