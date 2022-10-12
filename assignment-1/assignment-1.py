# r10142008 周昕妤 
# stream cloud link: 

from lib2to3.pgen2 import token
from numpy import disp, outer
import streamlit as st
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
import torch
import json
import glob
import re
import pandas as pd
from PIL import Image
from CwnGraph import CwnImage
from snownlp import SnowNLP
import seaborn as sns
import matplotlib.pyplot as plt

#from func import load_json, extract_content, ckipped_ner, ckipped_ws, ckipped_pos, cwn_tagged, snow_analyze, make_senti_plot

# models
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
    #plt.rcParams['font.sans-serif']=['Taipei Sans TC Beta']
    #plt.rcParams['axes.unicode_minus'] = False

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

def load_image(image_file):
	img = Image.open(image_file)
	return img


# ptt dataset
#horror_jsons = os.listdir("./data/Horror/2020")
#food_jsons = os.listdir("./data/Food/2020")
horror_jsons = glob.glob("./data/Horror/2020/*")
food_jsons = glob.glob("./data/Food/2020/*")


# start designing layout
st.set_page_config(layout="wide")
st.title('中文NLP管線處理')
st.caption('Streamlit project   r10142008 周昕妤')

menu = ['Food', 'Horror', 'img']
choice = st.sidebar.selectbox("PTT boards", menu)

if choice == 'Food':
    
    n_file = 0
    food_cont = []
    for food in food_jsons:
        #filenames = (f"./data/Food/2020/{food}")
        files = load_json(food)
        cont = extract_content(files)
        food_cont.append(food)
        n_file += 1
    st.success(f"Successfully load {n_file} posts from PTT Food Forum (2020)")

   # st.write('資料取自PTT Food版共7718篇貼文（2020）')
  #  step1, display1 = st.columns([1, 4])
   # with step1:
    st.markdown('#### 搜尋 Search')
    #with display1:
    with st.form(key='searchForm'):
        search_word = st.text_area('請輸入搜尋字詞')
        window = st.slider('要選擇多大的 window size?', 1, 10, 1)
        btn = st.form_submit_button(label='提交')
            
        if btn:
            with st.spinner('Loading models...'):
                warmup()
            step2, display2 = st.columns([1, 4])
            step3, display3 = st.columns([1, 4])
            step4, display4 = st.columns([1, 4])
            step5, display5 = st.columns([1, 4])
            step6, display6 = st.columns([1, 4])
                
            with step2:
                st.markdown('#### 相關文章')
            with display2:
                output = []
                for cont in food_cont:
                    for idx, con in enumerate(cont):
                        if re.search(search_word, con):
                            before = idx - window
                            after = idx + window + 1
                            output.append(cont[before:after])
                st.write(f'## 搜尋結果：共有{len(output)}筆搜尋結果。')
                for out in output:
                    st.write(out)

                
                                
            with step3:
                st.markdown('#### CKIP 斷詞系統 (Tokenization)')
            with display3:
                for out in output:
                    tokenized = ckipped_ws(out)
                    st.write(tokenized)
                
            with step4:
                st.markdown('#### CKIP 詞類標記 (Part-of-Speech Tagging)')
            with display4:
                for out in output:
                    tokenized = ckipped_ws(out)
                    pos = ckipped_pos(tokenized)
                    st.write(pos)

            with step5:
                st.markdown('#### CKIP 命名實體 (Named Entity Recognition)')
            with display5:
                ner_words, ner_types = [], []
                for out in output:
                    ner_word, ner_type = ckipped_ner(out)
                    ner_words.append(ner_word)
                    ner_types.append(ner_type)
                for ner_word in ner_words:
                    for i in range(len(ner_word)):
                        ent = (ner_word[i], ' (', ner_type[i], ')')
                        entj = ''.join(ent)
                        st.write(entj)

            with step6:
                st.markdown('#### CWN 詞意自動標記 (Sense Tagging)')
            with display6:
                st.write('to be continued...')
         #   word = tokenized.split(' ')
          # senses, num_of_sense = [], []
            #for w in word:
         #       s, ns = cwn_tagged(w)
             #   st.write('詞意數量:', ns)
              #  st.write('詞意條目:', s)

if choice == 'Horror':
    #st.write('資料取自PTT Horror版共194篇貼文（2020）')
    horror_cont = []
    for horror in horror_jsons:
        #filenames = (f"./data/Horror/2020/{horror}")
        files = load_json(horror)
        cont = extract_content(files)
        horror_cont.append(cont)
    st.success(f"Successfully load {len(horror_cont)} posts from PTT Horror Forum (2020)")

    c = st.container()
    with c:
        with st.form(key='searchForm'):
            search_word = st.text_input('請輸入搜尋字詞')
            window = st.slider('要選擇多大的 window size?', 1, 10, 1)
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

                    n = 1
                    for out in output[:5]:
                        if len(out) == 0:
                            pass
                        else:
                            st.write(f'### --------------------第{n}筆搜尋結果--------------------')
                            st.write(''.join(out))
                            checked = st.checkbox('情感分析 Sentiment Analysis (by sentence)', key=n)
                            if checked:
                                senti_df = snow_analyze(out)
                                st.table(senti_df)
                                make_senti_plot(senti_df)
                            n += 1
                    with st.expander('更多結果請按此查詢'):
                        for out in output[6:]:
                            if len(out) == 0:
                                pass
                            else:
                                st.write(f'--------------------第{n}筆搜尋結果--------------------')
                                st.write(''.join(out))
                                n += 1

                c3 = st.container()
                with c3:

                    st.markdown('## 情感分析 Sentiment Analysis (by sentence)')
                    num_post = st.slider("選擇欲分析的貼文：", 1, len(output), 1)

                    if num_post:
                        #visualized = st.checkbox('情感分析視覺化')
                        senti_df = snow_analyze(output[num_post-1])
                        st.table(senti_df)
                        make_senti_plot(senti_df)
                        




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