# r10142008 周昕妤 
# stream cloud link: 

from lib2to3.pgen2 import token
from numpy import disp, outer
import streamlit as st
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch
import json
import os
import re
import pandas as pd
from PIL import Image
from CwnGraph import CwnImage
from snownlp import SnowNLP
import seaborn as sns
import matplotlib.pyplot as plt
import DistilTag
from DistilTag import DistilTag
from collections import Counter

#from func import load_json, extract_content, ckipped_ner, ckipped_ws, ckipped_pos, cwn_tagged, snow_analyze, make_senti_plot

# models
ws_driver = None
pos_driver = None
ner_driver = None
cwn_tagger = None
distil_tagger = None

# Load ckip models
def ckip_warmup():
   global ws_driver, pos_driver, ner_driver

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

def cwn_warmup():
    global distil_tagger, cwn_tagger

    if cwn_tagger is None:
       cwn_tagger = CwnImage.latest()
       print('cwn tagger ready')
    if distil_tagger is None:
        #DistilTag.download()
        distil_tagger = DistilTag()
        print('distil tagger ready')

def load_json(file_path):            
    with open(file_path, 'r', encoding="UTF-8") as file:
        data = json.load(file)
    return data

def extract_title(json_data):
    title = [json_data['post_title']]
    return title

def extract_content(json_data): 
    # 把'post_body'跟'content'文字內容取出
    content = []
    content.append(json_data['post_body'])
    for i in range(len(json_data['comments'])):
        content.append(json_data['comments'][i]['content'])           
    # 清理圖片並以\n斷開句子
    for i in range(len(content)):
        content[i] = re.sub('http(s)?://.+.jpg', '', content[i])
        content[i] = re.sub('\n\n', '\n', content[i])
        content[i] = re.sub('--', '', content[i])
        content[i] = content[i].split('\n')   
    contents = [con for cont in content for con in cont]
    # Clean urls
    for con in contents:
        if re.search('^https?:\/\/.*[\r\n]*', con):
            contents.remove(con)
        if len(con) == 0:
            contents.remove(con)   
    return contents    

def snow_analyze(rawText):
    
    scores = []
    for text in rawText:
        res = SnowNLP(text)
        scores.append(res.sentiments)

    df = pd.DataFrame({
        'sentence':rawText,
        'score':scores
        })

    return df

def make_senti_plot(df):
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    fig = plt.figure(figsize=(20, 5))
    plt.title('Sentiment Scores Plot')
    plt.xticks(fontsize = 20, rotation = 90)
    sns.barplot( x = 'sentence', y = 'score', data = df, palette = "pastel")
    st.pyplot(fig)
    
# ckip tokenization
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

# distil tagger
def cwn_tagged(sent):
    tagged = distil_tagger.tag(sent)
    output = []
    for tag in tagged:
        for t in tag:
            out = t[0], ' (', t[1], ')'
            output.append(''.join(out))
    output = ' '.join(output)
    return output

def load_image(image_file):
	img = Image.open(image_file)
	return img


# ptt dataset
horror_jsons = os.listdir("assignment-1/data/Horror/2020")
food_jsons = os.listdir("assignment-1/data/Food/2020")

# start designing layout
st.set_page_config(layout="wide")
st.title('中文NLP管線處理：以批踢踢語料庫Food版和Horror版為例')
st.caption('Streamlit project   r10142008 周昕妤')

menu = ['Food', 'Horror', 'img']
choice = st.sidebar.selectbox("PTT Boards", menu)

if choice == 'Food':
    
    n_file = 0
    food_cont, food_title = [], []
    for food in food_jsons:
        filenames = (f"./data/Food/2020/{food}")
        files = load_json(filenames)
        cont = extract_content(files)
        titl = extract_title(files)
        food_cont.append(cont)
        food_title.append(titl)
        n_file += 1
    st.success(f"Successfully load {n_file} posts from PTT Food Forum (2020)")
    title_keys = []
    for title in food_title:
        for t in title:
            matched = re.search('\[.+\]', t)
            try:
                title_keys.append(matched[0])
            except:
                pass
    counter = Counter(title_keys).most_common(10)
    k, c = [], []
    for con in counter:
        k.append(con[0])
        c.append(con[1])
    kcdf = pd.DataFrame({
        'Keyword':k,
        'Count':c
    })
    st.markdown('### Top 10 Titles')
    st.table(kcdf)

    with st.form(key='test'):
        search_word = st.text_input('請輸入搜尋字詞（可根據上表排名搜尋相關美食資訊！） 例如：')
        window = st.slider('要選擇多大的 window size?', 5, 10, 1)
        buttn = st.form_submit_button(label='Search')
        if buttn:
            with st.spinner('Loading models...'):
                cwn_warmup()
            c2 = st.container()
            with c2:
                output = []
                for cont in food_cont:
                    for idx, con in enumerate(cont):
                        if re.search(search_word, con):
                            before = idx - window
                            after = idx + window + 1
                            output.append(cont[before:after])

                st.write(f'## 搜尋結果：共有{len(output)}筆搜尋結果。')

                example = output[0]
                more_info = output[1:]

                # print the firs post as example
                st.write('#### --------------------第1筆搜尋結果--------------------')
                st.markdown('### 斷詞及詞性標記 Tokenization and Part-of-Speech Tagging')
                for ex in example:
                    st.write(cwn_tagged(ex))
                st.markdown('### 逐句情感分析 Sentiment Analysis (by sentence)')
                senti_df = snow_analyze(example)
                st.table(senti_df)
                make_senti_plot(senti_df)

                n = 2
                with st.expander('更多結果請按此查詢'):
                    for out in output[1:]:
                        if len(out) == 0:
                            pass
                        else:
                            st.write(f'#### --------------------第{n}筆搜尋結果--------------------')
                            st.markdown('### 斷詞及詞性標記 Tokenization and Part-of-Speech Tagging')
                            try:
                                for o in out:
                                    st.write(cwn_tagged(o))
                            except:pass
                            st.markdown('### 情感分析 Sentiment Analysis (by sentence)')
                            senti_df = snow_analyze(out)
                            st.table(senti_df)
                            make_senti_plot(senti_df)

if choice == 'Horror':
    horror_cont, horror_title = [], []
    #for horror in list(horror_jsons.iterdir()):
      #  if horror.exists():
    for horror in horror_jsons:
        filenames = (f"./data/Horror/2020/{horror}")
        files = load_json(filenames)
        cont = extract_content(files)
        titl = extract_title(files)
        horror_cont.append(cont)
        horror_title.append(titl)
    st.success(f"Successfully load {len(horror_cont)} posts from PTT Horror Forum (2020)")

    title_keys = []
    for title in horror_title:
        for t in title:
            matched = re.search('\[.+\]', t)
            try:
                title_keys.append(matched[0])
            except:
                pass
    counter = Counter(title_keys).most_common(10)
    k, c = [], []
    for con in counter:
        k.append(con[0])
        c.append(con[1])
    kcdf = pd.DataFrame({
        'Keyword':k,
        'Count':c
    })
    st.markdown('### Top 10 Titles')
    st.table(kcdf)

    c = st.container()
    with c:
        with st.form(key='searchForm'):
            search_word = st.text_input('請輸入搜尋字詞')
            window = st.slider('要選擇多大的 window size?', 5, 10, 1)
            btn = st.form_submit_button(label='提交')

            if btn:
                c2 = st.container()
                with c2:
                    output = []
                    for cont in horror_cont:
                        for idx, con in enumerate(cont):
                            if re.search(search_word, con):
                                before = idx - window
                                after = idx + window + 1
                                output.append(cont[before:after])

                    st.write(f'## 搜尋結果：共有{len(output)}筆搜尋結果。')


                    example = output[0]
                    more_info = output[1:]

                    # print the firs post as example
                    st.write('#### --------------------第1筆搜尋結果--------------------')
                    st.markdown('### 斷詞及詞性標記 Tokenization and Part-of-Speech Tagging')
                    for ex in example:
                        st.write(cwn_tagged(ex))
                    st.markdown('### 逐句情感分析 Sentiment Analysis (by sentence)')
                    senti_df = snow_analyze(example)
                    st.table(senti_df)
                    make_senti_plot(senti_df)

                    n = 2
                    with st.expander('更多結果請按此查詢'):
                        for out in output[1:]:
                            if len(out) == 0:
                                pass
                            else:
                                st.write(f'#### --------------------第{n}筆搜尋結果--------------------')
                                st.markdown('### 斷詞及詞性標記 Tokenization and Part-of-Speech Tagging')
                                try:
                                    for o in out:
                                        st.write(cwn_tagged(o))
                                except:pass
                                st.markdown('### 情感分析 Sentiment Analysis (by sentence)')
                                senti_df = snow_analyze(out)
                                st.table(senti_df)
                                make_senti_plot(senti_df)
                                n += 1




    if choice == 'img':
        st.title("Image")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

        if image_file is not None:

            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                                "filesize":image_file.size}
            st.write(file_details)

            # To View Uploaded Image
            st.image(load_image(image_file),width=250)