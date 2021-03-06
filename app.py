from flask import Flask, render_template, request
from visualizations import *

import matplotlib.pyplot as pPlot
from wordcloud import WordCloud, STOPWORDS
import numpy as npy
from PIL import Image


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard', methods=['GET','POST'])
def createDashboard():
    
    # get text from form
    text = request.form
    for value in text.values():
        formText = value
    
    # create and save wordcloud
    wc_byte = create_word_cloud(formText)

    # create word frequency table
    table = word_freq_table(formText)

    # create word frequency bar chart
    bar_chart = word_freq_bar(formText)

    # generate summary with default options
    summ = generate_summary(formText)

    # pos tagging
    nouns, verbs, adj = pos_tags(formText)

    word_count, sent_count = word_and_sent_count(formText)

    tone, context = tone_and_context(formText)

    language = lang_detection(formText)

    return render_template('dashboard.html',text = formText, wc = wc_byte, table = table, 
                            bar_chart = bar_chart, summ = summ,
                            nouns = nouns, verbs = verbs, adj = adj,
                            word_count = word_count, sent_count = sent_count,
                            tone = tone, context = context, language = language)

if __name__ == "__main__":
    app.run(debug=True)