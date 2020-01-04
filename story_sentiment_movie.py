from bs4 import BeautifulSoup
import requests
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url_little_women        = 'http://www.gutenberg.org/files/514/514-h/514-h.htm'
url_war_and_peace       = 'http://www.gutenberg.org/files/2600/2600-h/2600-h.htm'
url_bible               = 'http://www.gutenberg.org/files/10/10-h/10-h.htm'
url_wizard_of_oz        = 'http://www.gutenberg.org/files/55/55-h/55-h.htm'
url_les_mis             = 'http://www.gutenberg.org/files/135/135-h/135-h.htm'
url_alice_in_wonderland = 'http://www.gutenberg.org/files/11/11-h/11-h.htm'
url_tom_sawyer          = 'http://www.gutenberg.org/files/74/74-h/74-h.htm'
url_peter_pan           = 'http://www.gutenberg.org/files/16/16-h/16-h.htm'
url_moby_dick           = 'http://www.gutenberg.org/files/2701/2701-h/2701-h.htm' 
url_don_quixote         = 'http://www.gutenberg.org/files/996/996-h/996-h.htm'

def sentiment_main(url, title, window = False):
    
    tokenized_text = preprocess_text(url)
    df_sentiment = sentiment_analysis(tokenized_text)
    
    
    ax = sentiment_plot(df_sentiment, title, window)
    plt.show()
    
def preprocess_text(url):
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
    return tokenized_text

def sentiment_analysis(tokenized_text):
    sia     = SIA()
    results = []
    
    for line in tokenized_text:
        pol_score             = sia.polarity_scores(line)
        pol_score['sentence'] = line
        results.append(pol_score)
    df_sentiment = pd.DataFrame.from_records(results)
    return df_sentiment
    

def sentiment_plot(df_sentiment, title, window):

    rolling_window = int(df_sentiment.shape[0] / 20)                 
    df_sentiment["rolling_pos"] = df_sentiment["pos"].rolling(rolling_window).mean()
    df_sentiment["rolling_neg"] = df_sentiment["neg"].rolling(rolling_window).mean()
    
    c = np.linspace(0, 1, df_sentiment.shape[0])
    
    cmap = sns.cubehelix_palette(8, dark=0.1, light=0.9, as_cmap=True)
    #ax = plt.plot(df_sentiment["rolling_pos"], df_sentiment["rolling_neg"], '-', color='blue', lw=0.8, label = 'Sentiment')
    ax = sns.scatterplot(x = df_sentiment["rolling_pos"], y = df_sentiment["rolling_neg"], hue = c, palette = cmap, alpha = 0.5)
    plt.legend(loc='upper right', labels=['Start', '1/3', '2/3', 'End'])
    #plt.xlabel('Positivity')
    #plt.ylabel('Negativity')
    #plt.title(title)
    #plt.legend()
        
    return ax
sentiment_main(url_peter_pan, "Peter Pan")
