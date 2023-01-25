import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

word2idx=dict(pd.read_csv('word2idx.csv').set_index('index').iloc[:,0])
tag2idx=dict(pd.read_csv('tag2idx.csv').set_index('index').iloc[:,0])
model = pickle.load(open('model.pkl','rb'))
st.title('ENITTY DETECTOR FROM ESSAYS AND ARTICLES')


# text='British demonstrators marched through that country'


def main():
    text=st.text_area("Please write an article below",max_chars=50000)
    
    
    butt = st.button('SUBMIT')
    if butt:
        sentence_list=str(text).split(' ')
        
        # for word in sentence_list:
        #     if word not in list(word2idx.keys()):
        #         word2idx[word]=len(list(word2idx.keys()))+1
        length=len(list(word2idx.keys()))
        lisst=[word2idx[elem] for elem in sentence_list]
        l=pad_sequences(maxlen=50, sequences=[lisst], padding='post',value=length-1)
        pred=model.predict(np.array([l[0]]))
        pred=np.argmax(pred, axis=-1)
        
        ip_sen_list=[]
        for elem in l[0]:
            for key,values in word2idx.items():
                if values==elem:
                    ip_sen_list.append(key)
                
        ip_tag_list=[]            
        for elem in pred[0]:
            for key,values in tag2idx.items():
                if values==elem:
                    ip_tag_list.append(key)
        output_dict={}           
        for elem in range(0,len(lisst)):
            output_dict[ip_sen_list[elem]]=ip_tag_list[elem]
        st.json(output_dict)

if __name__=='__main__':
    main()