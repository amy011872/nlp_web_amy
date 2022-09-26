# r10142008 周昕妤 
# stream cloud link: https://amy011872-nlp-web-amy-assignment-bonus-1-z6ldel.streamlitapp.com/

#import time
#from click import progressbar
import streamlit as st
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from snownlp import SnowNLP


def vader_anaylze(rawText):
    anaylzer = SentimentIntensityAnalyzer()
    compound = anaylzer.polarity_scores(rawText)['compound']
    pos_list, neg_list, neu_list = [], [], []
    for i in rawText.split():
        senti = anaylzer.polarity_scores(i)['compound']
        if senti > 0:
            pos_list.append((i, senti))
        elif senti < 0:
            neg_list.append((i, senti))
        else:
            neu_list.append((i, senti))

    return pos_list, neg_list, neu_list, compound

def snow_analyze(rawText):
    res = SnowNLP(rawText)
    senti_score = res.sentiments

    return senti_score

st.markdown('# Welcome to my sentiment anaylsis app!')
menu = ['English version', 'Chinese version']
choice = st.sidebar.selectbox("Language", menu)

if choice == 'English version':
    col1, col2 = st.columns(2)

    with col1.form(key='snetiForm'):
        raw_text = st.text_area('Please enter a sentence here:')
        btn = st.form_submit_button(label='Ａnalyze')

        if btn:
      #  progress_bar = st.progress(0)
       # for percent in range(100):
        #    time.sleep(0.01)
         #   progress_bar.progress(percent + 1)

            with col2:
                st.success("Finish analyzing!")
                st.markdown('### Results from different models:')

                polarity = TextBlob(raw_text).sentiment.polarity
                if polarity > 0:
                    st.markdown('Textblob: Positive 🥰')
                elif polarity < 0:
                    st.markdown('Textblob: Negative 😭')
                else:
                    st.markdown('Textblob: Neutral 😶')    

                pos_list, neg_list, neu_list, compound = vader_anaylze(raw_text)
                if compound > 0:
                    st.markdown('vaderSentiment: Positive 🥰')
                elif compound < 0:
                    st.markdown('vaderSentiment: Negative 😭')
                else:
                    st.markdown('vaderSentiment: Neutral 😶') 

           
    if len(raw_text) != 0:
        with st.expander('Click here to have more details of the analysis.'):
            st.markdown('#### Results from textblob:')
            polarity = TextBlob(raw_text).sentiment.polarity
            subjectivity = TextBlob(raw_text).sentiment.subjectivity
            st.write('Polarity:', polarity)
            st.write('Subjectivity:', subjectivity)
            if polarity > 0:
                st.markdown('Sentiment: Positive 🥰')
            elif polarity < 0:
                st.markdown('Sentiment: Negative 😭')
            else:
                st.markdown('Sentiment: Neutral 😶')   
    
            st.markdown('#### Results from vaderSentiment:')
            pos_list, neg_list, neu_list, compound = vader_anaylze(raw_text)
            if compound > 0:
                st.markdown('Sentiment: Positive 🥰')
            elif compound < 0:
                st.markdown('Sentiment: Negative 😭')
            else:
                st.markdown('Sentiment: Neutral 😶') 
            pos_col, neu_col, neg_col = st.columns(3)
            pos_col.write('Positive tokens:')
            for pos in pos_list:
                pos_col.write(pos)
            neu_col.write('Neutral tokens:')
            for neu in neu_list:
                neu_col.write(neu)
            neg_col.write('Negative tokens:')
            for neg in neg_list:
                neg_col.write(neg)

if choice == 'Chinese version':

    st.markdown('#### 歡迎使用中文版本！')
    
    col1, col2 = st.columns(2)

    with col1.form(key='snetiForm'):
        raw_text = st.text_area('請輸入一個句子')
        btn = st.form_submit_button(label='分析')

        if btn:
            with col2:
                st.success("Finish analyzing!")
                senti_score = snow_analyze(raw_text)
                st.write('分數：', senti_score)
                if senti_score > 0.5:
                    st.write('Sentiment: 正向！🥰')
                elif senti_score < 0.5:
                    st.write('Sentiment: 負向 😭😭😭')
                else:
                    st.write('Sentiment: 中性 😶😶')
                