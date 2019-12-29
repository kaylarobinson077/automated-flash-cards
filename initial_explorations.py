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
    sia     = SIA()
    results = []
    
    for line in tokenized_text:
        pol_score             = sia.polarity_scores(line)
        pol_score['sentence'] = line
        results.append(pol_score)
    df_sentiment = pd.DataFrame.from_records(results)
    return df_sentiment



def label_chapters(df_sentiment):
    # make df with page ranges per chapter
    chapter_breaks                        = df_sentiment[df_sentiment['sentence'].str.contains("CHAPTER")].index
    df_chapter_boundaries                 = pd.DataFrame(columns=['ch_start', 'ch_end'])
    df_chapter_boundaries['ch_start']     = chapter_breaks
    df_chapter_boundaries['ch_end'][0:-1] = chapter_breaks[1:]-1
    df_chapter_boundaries['ch_end'][-1:]  = df_sentiment.shape[0]
    
    # get rid of one-page chapters, because these are most likely the table of contents
    for index, row in df_chapter_boundaries.iterrows():
        ch_start = row['ch_start']
        ch_end   = row['ch_end']
        
        if ch_end - ch_start < 2:
            df_chapter_boundaries.drop(index, inplace = True)
    df_chapter_boundaries.reset_index(inplace = True)
            
    print(df_chapter_boundaries.head())
    
    # add column to df_sentiment for chapter number
    df_sentiment['chapter'] = np.nan
    df_sentiment['chapter_frac'] = np.nan # will be x ticks in plot
    for index, row in df_chapter_boundaries.iterrows():
        ch_start = row['ch_start']
        ch_end   = row['ch_end']
        df_sentiment['chapter'][ch_start:ch_end+1] = index + 1 # python starts counting at zero
        
        n_sentences = df_sentiment[ch_start:ch_end+1].shape[0]
        df_sentiment['chapter_frac'][ch_start:ch_end+1] = np.linspace(0, 1, n_sentences)
    
    return df_sentiment
    

def sentiment_plot(df_sentiment, title, window):

    if window == False:
        rolling_window = int(df_sentiment.shape[0] / 20)
        
    if window == 'chapter':
        df_sentiment = label_chapters(df_sentiment) # add column with chapters
        rolling_window = int(df_sentiment.shape[0] / df_sentiment['chapter'].max())
        
    df_sentiment["rolling_pos"] = df_sentiment["pos"].rolling(rolling_window).mean()
    df_sentiment["rolling_neg"] = df_sentiment["neg"].rolling(rolling_window).mean()
        
    if window != 'chapter':

        x = np.linspace(1,df_sentiment.shape[0], df_sentiment.shape[0])
        ax = plt.plot(x, df_sentiment["rolling_pos"], '-', color='green', lw=0.8, label = 'Positive')
        ax = plt.plot(x, df_sentiment["rolling_neg"], '-', color='red', lw=0.8, label = 'Negative')
        plt.xlabel('Sentence Number')
        plt.ylabel('Fraction')
        plt.title(title)
        plt.legend()
        
    if window == 'chapter':
        
        fig, ax = plt.subplots(figsize=(10,8))
        classes = list(df_sentiment.chapter.dropna().unique())
        
        #colors = plt.cm.cool(np.linspace(0,1,max(classes)))
        for c in classes:
            #print(c)
            #print(int(c))
            df2 = df_sentiment.loc[df_sentiment['chapter'] == c]
            df2.plot(x='chapter_frac', y='rolling_pos', ax=ax, label=c)
            #df2.plot(x='chapter_frac', y='rolling_pos', ax=ax, label=c, color=colors[int(c)] )
        
        ax.get_legend().remove()
        
    return ax



# let's run it!
sentiment_main(url_little_women, "Little Women")
sentiment_main(url_war_and_peace, "War and Peace")
sentiment_main(url_bible, "The Bible")
sentiment_main(url_wizard_of_oz, "The Wizard of Oz")
sentiment_main(url_les_mis, "Les Miserables")
sentiment_main(url_alice_in_wonderland, "Alice in Wonderland")