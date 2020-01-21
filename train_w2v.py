import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import matplotlib.pyplot as plt
import jieba
import jieba.analyse
from gensim.models import word2vec
import gensim
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.corpora import LowCorpus
from gensim.corpora import Dictionary
import re
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping
import pandas as pd

if __name__ == '__main__':
    a={'name': '郭 韡', 'type': 'per'}
    print(a['type'])
    data=pd.read_csv("./data/train.csv",encoding="utf-8")
    test=pd.read_csv("./data/test_stage1.csv",encoding="utf-8")
    train_data=[i for i in list(data["text"])]
    test_data=[i for i in list(test['text'])]
    sentence = train_data+test_data
    sen=[list(jieba.cut(x)) for x in sentence]
    model = word2vec.Word2Vec(sen, size=256, window=2, min_count=1, workers=4)
    model.save("./model/text_w2v.model")
    model.wv.save_word2vec_format('./model/vkmodel.txt', binary=False)
    print("字向量模型训练完毕")
    print(1)
