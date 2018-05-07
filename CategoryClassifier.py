#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator
from TextNormalizer import TextNormalizer
from sklearn.pipeline import Pipeline
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from keras.models import Sequential
from keras.layers.convolutional import Conv1D,MaxPooling1D,AveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint


class UnknownModelOption(Exception):
    pass 
class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, mtype):
        def load_gb_vectors():
            self.__w2v_sg = pickle.load(open('w2v_sg.pkl','rb'))
            self.__w2v_cbw = pickle.load(open('w2v_cbw.pkl','rb'))
        def load_net_vectors():
            self.__w2v_sg = pickle.load(open('w2v_sg.pkl','rb'))
        load_options = {
                'GB': load_gb_vectors(),
                'NET': load_net_vectors()
                }
        load_options[mtype]
        self.__mtype = mtype

    def __transform_NET(self, X):
        def make_padded_sequenses(docs, max_length,w2v):
            tokens = [doc.split(' ') for doc in docs[0]]
            vecs = [[w2v[t] if t in w2v else np.zeros(50) for t in ts] for ts in tokens]
            seqs = np.array([np.pad(np.vstack(v),mode = 'constant', pad_width = ((0,max_length-len(v)),(0,0))) if len(v)<max_length else np.vstack(v)[:max_length,:] for v in vecs])
            return seqs
        vector_sg = make_padded_sequenses([X], 15,w2v = self.__w2v_sg)
        return vector_sg
        
    def __transform_GB(self, X):
        def make_avg_sequenses(docs,w2v):
            tokens = [doc.split(' ') for doc in docs]
            vecs = [[w2v[t] if t in w2v else np.zeros(50) for t in ts] for ts in tokens]
            seqs = np.array([np.apply_along_axis(np.mean, 0, np.vstack(v)) for v in vecs])
            return seqs
        vector_sg = make_avg_sequenses(X,w2v = self.__w2v_sg)
        vector_cbw = make_avg_sequenses(X,w2v = self.__w2v_cbw)
        vector = np.hstack([vector_sg,vector_cbw])
        return vector.reshape(1,-1)
    
    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        transform_options = {
                'GB': self.__transform_GB(X),
                'NET': self.__transform_NET(X)
                }
        return transform_options[self.__mtype]

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

class QuestionClassifier():
    
    def __init__(self, mtype):
        
        self.__tn = TextNormalizer().fit()
        self.__vectorizer = Vectorizer(mtype).fit()
        self.__load_model(mtype)
        self.__mtype = mtype
        
    def __prepare_network(self):
        model = Sequential()
        model.add(Conv1D(filters = 25, kernel_size = 2, input_shape = (15, 50), activation='relu'))
        model.add(Conv1D(filters = 20, kernel_size = 2, activation='relu'))
        model.add(Conv1D(filters = 15, kernel_size = 2, activation='relu'))
        model.add(Conv1D(filters = 10, kernel_size = 2, activation='relu'))
        model.add(MaxPooling1D(1))
        model.add(Flatten())
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        model.load_weights('weights.hdf5')
        return model

    def __load_model(self,mtype):
        load_options = {
                'GB': pickle.load(open('GBoost.pkl','rb')),
                'NET': self.__prepare_network()
                }
        if mtype in load_options:
            self.__model = load_options[mtype]
        else:
            raise UnknownModelOption('Specified model option is not supprted')

    def predict(self, text):
        class_names = ['другие',
            'вопросы о сервисе',
            'вопросы о товаре',
            'общеразговорные',
            'более 1 категории']

        cleaned_text = self.__tn.transform([text.lower()])
        feats = self.__vectorizer.transform(cleaned_text)
        prediction = self.__model.predict(feats)
        model_out = {
        'GB': prediction[0],
        'NET': np.argmax(prediction)
        }
        class_num = model_out[self.__mtype]
        return class_names[class_num]
    

if __name__ == '__main__':
    classifier = QuestionClassifier(mtype = 'NET')
    while True:
        review = input()
        print('\n')
        print(classifier.predict(review))