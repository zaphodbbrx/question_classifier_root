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

class UnknownModelOption(Exception):
    pass 
class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self, mtype):
        def load_gb_vectors():
            self.__w2v_sg = pickle.load(open('w2v_sg.pkl','rb'))
            self.__w2v_cbw = pickle.load(open('w2v_cbw.pkl','rb'))
        load_options = {
                'GB': load_gb_vectors()
                }
        load_options[mtype]
        self.__mtype = mtype

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
                'GB': self.__transform_GB(X)
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

    def __load_model(self,mtype):
        load_options = {
                'GB': pickle.load(open('GBoost.pkl','rb'))
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
        return class_names[prediction[0]]
if __name__ == '__main__':
    classifier = QuestionClassifier(mtype = 'GB')
    while True:
        review = input()
        print('\n')
        print(classifier.predict(review))