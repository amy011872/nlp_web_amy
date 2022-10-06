# r10142008 周昕妤 
# stream cloud link: 

from lib2to3.pgen2 import token
from numpy import disp
import streamlit as st
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch
import re
import pandas as pd
from PIL import Image
from CwnGraph import CwnImage

st. set_page_config(layout="wide")
st.title('中文NLP管線處理')
st.caption('Streamlit project   r10142008 周昕妤')



# Text pipeline: tokenization, pos tagging, named entity recognition, sense tagging, sentiment analysis, 

ws_driver = None
pos_driver = None
ner_driver = None
cwn_tagger = None

# Load ckip models
def warmup():
   global ws_driver, pos_driver, ner_driver, cwn_tagger

   if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        if ws_driver is None:
            ws_driver = CkipWordSegmenter(device=device)
            print('ws driver ready')
        if pos_driver is None:
            pos_driver = CkipPosTagger(device=device)
            print('pos driver ready')
        if ner_driver is None:
            ner_driver = CkipNerChunker(device=device)
            print('ner driver ready')
   if cwn_tagger is None:
       cwn_tagger = CwnImage.latest()
       print('cwn tagger ready')

# Load ptt dataset
def load_ppt():
    #testing
    global titles
    df = pd.read_csv('../../nlp_web/assignments/ptt-crawler/data/Soft_job/2019/Soft_job_2019_10.csv')
    titles = df.title.to_list()

    return titles

# Search
def search_ptt(target, search_word):
    if target == 'title':
        for t in titles:
            if re.search(search_word, t):
                print(t)

# ckip tokenziation 
def ckipped_ws(input):
    input = [input]
    ws = ws_driver(input, use_delim = False)
    return ' '.join(ws[0])

# ckip pos tagging
def ckipped_pos(input):
    input = [input]
    ws = ws_driver(input, use_delim = False)
    pos = pos_driver(ws, use_delim = False)

    assert len(ws) == len(pos)
    res = []
    for word_ws, word_pos in zip(ws, pos):
       for wws, wpos in zip(word_ws, word_pos):
         res.append((wws,wpos))

    wp = [' '.join(r) for r in res]
    return '    '.join(wp)

# ckip ner
def ckipped_ner(input):
    input = [input]
    ner = ner_driver(input, use_delim=False)
    ner_word = [i[0] for n in ner for i in n]
    ner_type = [i[1] for n in ner for i in n]
    return ner_word, ner_type

# cwn sense tagger (CwnGraph)
def cwn_tagged(lemma):
    if cwn_tagger is None:
       print('re-initializing ckip...')
       warmup()
    
    tagged = cwn_tagger.find_lemma(lemma)
    if len(tagged) > 0:
        senses = tagged[0].senses
        num_of_sense = len(senses)
        return senses, num_of_sense

    else:
        return('No results')

# Upload image (for what??) 
def load_image(image_file):
	img = Image.open(image_file)
	return img







# start designing layout
menu = ['Text', 'Image']
choice = st.sidebar.selectbox("Modality", menu)

if choice == 'Text':
    warmup()

    step1, display1 = st.columns([1, 4])
    with step1:
        st.markdown('#### 搜尋 Search')
    with display1:
        with st.form(key='snetiForm'):
            input_sent = st.text_area('請輸入一句話')
            btn = st.form_submit_button(label='提交')
        with st.form(key='searchPTT'):
            titles = load_ptt()
            search_title = st.text_area('請輸入文章標題')
            btn_title = st.form_submit_button(label='搜尋')
    if btn_title:
        st.write(search_ptt('title', search_title))
    
    if btn:
        step2, display2 = st.columns([1, 4])
        step3, display3 = st.columns([1, 4])
        step4, display4 = st.columns([1, 4])
        step5, display5 = st.columns([1, 4])
        
        with step2:
            st.markdown('#### CKIP 斷詞系統 (Tokenization)')
        with display2:
            tokenized = ckipped_ws(input_sent)
            pos = ckipped_pos(tokenized)
            ner_word, ner_type = ckipped_ner(input_sent)
            st.write(tokenized)
        
        with step3:
            st.markdown('#### CKIP 詞類標記 (Part-of-Speech Tagging)')
        with display3:
            st.write(pos)

        with step4:
            st.markdown('#### CKIP 命名實體 (Named Entity Recognition)')
        with display4:
            for i in range(len(ner_word)):
                ent = (ner_word[i], ' (', ner_type[i], ')')
                entj = ''.join(ent)
                st.write(entj)

        with step5:
             st.markdown('#### CWN 詞意自動標記 (Sense Tagging)')
        with display5:
            word = tokenized.split(' ')
            senses, num_of_sense = [], []
            for w in word:
                s, ns = cwn_tagged(w)
                st.write('詞意數量:', ns)
                st.write('詞意條目:', s)



if choice == 'Image':
    st.title("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:

	    # To See details
	    file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
	    st.write(file_details)

        # To View Uploaded Image
	    st.image(load_image(image_file),width=250)