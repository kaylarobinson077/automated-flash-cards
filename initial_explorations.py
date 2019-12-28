# read in text, break into paragraphs (or sentences?)
from bs4 import BeautifulSoup
import requests
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url_little_women = 'http://www.gutenberg.org/files/514/514-h/514-h.htm'
url_war_and_peace = 'http://www.gutenberg.org/files/2600/2600-h/2600-h.htm'
url_bible = 'http://www.gutenberg.org/files/10/10-h/10-h.htm'
url_wizard_of_oz = 'http://www.gutenberg.org/files/55/55-h/55-h.htm'
url_les_mis = 'http://www.gutenberg.org/files/135/135-h/135-h.htm'
url_alice_in_wonderland = 'http://www.gutenberg.org/files/11/11-h/11-h.htm'

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
    sia = SIA()
    results = []
    
    for line in tokenized_text:
        pol_score = sia.polarity_scores(line)
        pol_score['sentence'] = line
        results.append(pol_score)
    df_sentiment = pd.DataFrame.from_records(results)
    return df_sentiment



def label_chapters(df_sentiment):
    chapter_breaks = df_sentiment[df_sentiment['sentence'].str.contains("CHAPTER")].index
    df_chapter_boundaries = pd.DataFrame(columns=['ch_start', 'ch_end'])
    df_chapter_boundaries['ch_start'] = chapter_breaks
    df_chapter_boundaries['ch_end'][0:-1] = chapter_breaks[1:]-1
    df_chapter_boundaries['ch_end'][-1:] = df_sentiment.shape[0]
    
    

def sentiment_plot(df_sentiment, title, window):
    #if window == 'chapter':
        
            
    if window == False:
        rolling_window = int(df_sentiment.shape[0] / 20)
        
    if window != 'chapter':
        df_sentiment["rolling_pos"] = df_sentiment["pos"].rolling(rolling_window).mean()
        df_sentiment["rolling_neg"] = df_sentiment["neg"].rolling(rolling_window).mean()
        #df_sentiment["rolling_neu"] = df_sentiment["neu"].rolling(rolling_window).mean()
        
        x = np.linspace(1,df_sentiment.shape[0], df_sentiment.shape[0])
        #y = df["rolling"]
        ax = plt.plot(x, df_sentiment["rolling_pos"], '-', color='green', lw=0.8, label = 'Positive')
        ax = plt.plot(x, df_sentiment["rolling_neg"], '-', color='red', lw=0.8, label = 'Negative')
        #plt.plot(x, df["rolling_neu"], '-', color='blue', lw=0.8)
        #plt.plot(x, df["rolling_pos"]-df["rolling_neg"], '-', color='black', lw=0.8)
        plt.xlabel('Sentence Number')
        plt.ylabel('Fraction')
        plt.title(title)
        plt.legend()
        
    if window == 'chapter':
        
        
    return ax



# let's run it!
sentiment_main(url_little_women, "Little Women")
sentiment_main(url_war_and_peace, "War and Peace")
sentiment_main(url_bible, "The Bible")
sentiment_main(url_wizard_of_oz, "The Wizard of Oz")
sentiment_main(url_les_mis, "Les Miserables")
sentiment_main(url_alice_in_wonderland, "Alice in Wonderland")