# read in text, break into paragraphs (or sentences?)
from bs4 import BeautifulSoup
import requests
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# open source text for Little Women
url = 'http://www.gutenberg.org/files/514/514-h/514-h.htm'
res = requests.get(url)
html_page = res.content
soup = BeautifulSoup(html_page, 'html.parser')

raw_text = soup.body.text

# process the text
processed_text = raw_text
processed_text = processed_text.replace('\r\n', ' ')
processed_text = re.sub('\n+', '\n', processed_text)

# tokenize at the sentence level
tokenized_text = sent_tokenize(processed_text)

# perform analysis at the sentence level
sia = SIA()
results = []

for line in tokenized_text:
    pol_score = sia.polarity_scores(line)
    pol_score['sentence'] = line
    results.append(pol_score)

df = pd.DataFrame.from_records(results)

# very busy plot of positivity per sentence
#x = np.linspace(1,df.shape[0], df.shape[0])
#y = df.pos
#plt.plot(x, y, 'o', color='black');

# let's look at rolling average of positivity
# circle back to think about the best number of sentneces over which to roll the average
# add vertical lines at our guesses for chapter breaks
chapter_breaks = df[df['sentence'].str.contains("CHAPTER")].index
for xc in chapter_breaks:
    plt.axvline(x=xc, ls='-', lw=0.3, color = 'grey')
    
rolling_window = 500

df["rolling_pos"] = df["pos"].rolling(rolling_window).mean()
df["rolling_neg"] = df["neg"].rolling(rolling_window).mean()
df["rolling_neu"] = df["neu"].rolling(rolling_window).mean()

x = np.linspace(1,df.shape[0], df.shape[0])
#y = df["rolling"]
l1 = plt.plot(x, df["rolling_pos"], '-', color='green', lw=0.8, label = 'Positive')
l2 = plt.plot(x, df["rolling_neg"], '-', color='red', lw=0.8, label = 'Negative')
#plt.plot(x, df["rolling_neu"], '-', color='blue', lw=0.8)
#plt.plot(x, df["rolling_pos"]-df["rolling_neg"], '-', color='black', lw=0.8)
plt.xlabel('Sentece Number')
plt.ylabel('Fraction')
plt.title('Little Women Sentiment (Rolling Averages)')
plt.legend()

