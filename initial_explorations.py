# read in text, break into paragraphs (or sentences?)
from bs4 import BeautifulSoup
import requests
import re

# open source text for Little Women
url = 'http://www.gutenberg.org/files/514/514-h/514-h.htm'
res = requests.get(url)
html_page = res.content
soup = BeautifulSoup(html_page, 'html.parser')
#text = soup.find_all(text=True)

raw_text = soup.body.text

# the text uses \r\n when text flows onto new line, clean this up
processed_text = raw_text.replace('\r\n', ' ')
# they left in escape sequence for '
#processed_text = processed_text.replace("\'", "'")
# if there are multiple newlines, replace with a single one
processed_text = re.sub('\n+', '\n', processed_text)

# split the processed text at new lines
paragraphs = processed_text.split('\n')

