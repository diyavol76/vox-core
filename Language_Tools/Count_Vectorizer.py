import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer ,PorterStemmer
from nltk.corpus import wordnet

nltk.download("wordnet")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def train_test(X_train,Y_train,X_test,Y_test):
    model=MultinomialNB()
    model.fit(X_train,Y_train)
    print("train score : ",model.score(X_train,Y_train))
    print("test score : ",model.score(X_test,Y_test))


def Vectorize(inputs_train,inputs_test,stop_words=None,tokenizer=None):

    vectorizer = CountVectorizer(stop_words=stop_words,tokenizer=tokenizer)
    Xtrain = vectorizer.fit_transform(inputs_train)
    Xtest = vectorizer.transform(inputs_test)
    return Xtrain,Xtest

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        tokens = word_tokenize(doc)
        words_and_tags= nltk.pos_tag(tokens)
        return [self.wnl.lemmatize(word,pos=get_wordnet_pos(tag))\
                for word, tag in words_and_tags]

class StemTokenizer:
    def __init__(self):
        self.porter = PorterStemmer()
    def __call__(self, doc):
        tokens= word_tokenize(doc)
        return [self.porter.stem(t) for t in tokens]

def simple_tokenizer(s):
    return s.split()


if __name__ == '__main__':



    df=pd.read_csv('bbc_text_cls.csv')

    inputs=df['text']
    labels=df['labels']

    #labels.hist(figsize=(10,5))
    #plt.show()

    #print(df.head())

    inputs_train,inputs_test,Ytrain,Ytest = train_test_split(inputs,labels,random_state=123)

    Xtrain,Xtest= Vectorize(inputs_train,inputs_test)

    #print(Xtrain)
    print("percentage of values are non-zero : " ,(Xtrain != 0).sum()/np.prod(Xtrain.shape))
    print("Try without nothing")
    print("Xtrain shape : " ,Xtrain.shape)
    train_test(Xtrain,Ytrain,Xtest,Ytest)

    #TRY with stop_words 'english'
    print("Try stop words")
    print("Xtrain shape : ", Xtrain.shape)
    Xtrain, Xtest = Vectorize(inputs_train, inputs_test,stop_words='english')
    train_test(Xtrain, Ytrain, Xtest, Ytest)

    #TRY with lemmatization
    print("Try lemmatization")
    print("Xtrain shape : ", Xtrain.shape)
    Xtrain,Xtest = Vectorize(inputs_train,inputs_test,tokenizer=LemmaTokenizer())
    train_test(Xtrain, Ytrain, Xtest, Ytest)

    #TRY with Stemming
    print("Try stemming")
    print("Xtrain shape : ", Xtrain.shape)
    Xtrain,Xtest = Vectorize(inputs_train,inputs_test,tokenizer=StemTokenizer())
    train_test(Xtrain, Ytrain, Xtest, Ytest)

    #TRY with simple tokenizer with split
    print("Try simple split tokenizer")
    print("Xtrain shape : ", Xtrain.shape)
    Xtrain,Xtest = Vectorize(inputs_train,inputs_test,tokenizer=simple_tokenizer)
    train_test(Xtrain, Ytrain, Xtest, Ytest)


