
import nltk
nltk.download('stopwords')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import numpy as np
from PIL import Image
import io
import urllib, base64
import collections
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize


def create_word_cloud(string):

    string  = string.lower()
    cloud = WordCloud(background_color = "white", max_words = 100, stopwords = set(STOPWORDS))
    wc = cloud.generate(string)
    # cloud.to_file("static/wordCloud.png")
    # print(type(wc))
    plt.imshow(wc, interpolation='bilinear')
    # plt.clf()
    plt.axis("off")
    plt.savefig('static/wordCloud.png', format = 'png')

    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)  # rewind the data
    string = base64.b64encode(image.read())

    # image_64 = 'data:image/png;base64,' + urllib.parse.quote(string)
    image_string = string.decode('utf-8')
    return image_string

# print(create_word_cloud("Test cloud"))

def clean_text(string):
    tokens = word_tokenize(string)
    words = [word for word in tokens if word.isalpha()]
    words = [word for word in tokens if len(word) > 2]
    words = [each_string.lower() for each_string in words]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]

    lem = WordNetLemmatizer()
    lemmatized_output = ' '.join([lem.lemmatize(w) for w in words])
    words = word_tokenize(lemmatized_output)
    return words

def word_freq_table(string):
    words = clean_text(string)
    count_of_words = collections.Counter(words)
    # top_10_words = count_of_words(10)

    df = pd.DataFrame(count_of_words.most_common(),columns = ['Word','Count'])
    df['Weightage'] = df['Count'] / sum(df['Count'])
    return df

def word_freq_bar(string):
    df = word_freq_table(string)
    # fig, ax = plt.plot(figsize=(10,7))
    fig = plt.figure(figsize =(20, 15)) 
    words = df['Word'].head(10)
    count = df['Count'].head(10)
    plt.barh(words[0:10], count[0:10]) 
    plt.xticks(fontsize=20, fontname='monospace')
    plt.yticks(fontsize=20, fontname='monospace')
    # plt.xlabel('Count')
    # plt.ylabel('Words')
    plt.legend('Count')
    plt.show()
    
    image = io.BytesIO()
    plt.savefig(image, format='png')
    image.seek(0)  # rewind the data
    string = base64.b64encode(image.read())

    # image_64 = 'data:image/png;base64,' + urllib.parse.quote(string)
    image_string = string.decode('utf-8')
    return image_string

def generate_summary(string):
    return summarize(string)

def pos_tags(string):
    df = word_freq_table(string)
    tags = nltk.pos_tag(df['Word'])
    
    noun_count = 0
    verb_count = 0
    adj_count = 0

    for word,tag in tags:
        
        df['POS tags'] = tag

        if tag == "NN" or tag == 'NNS':
            noun_count += 1
        elif tag == "VB" or tag == 'VBD':
            verb_count += 1
        elif tag == "JJ" or tag == 'JJR' or tag == 'JJS':
            adj_count += 1
    
    return noun_count, verb_count, adj_count

def word_and_sent_count(string):
    
   word_count = len(word_tokenize(string))
   sent_count = len(sent_tokenize(string))

   return word_count, sent_count
