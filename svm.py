import tarfile
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import re
import pandas as pd 
import json 
import multiprocessing
from collections import Counter
import statistics
from operator import itemgetter 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import gensim
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier


def tokenize(text):
	'''Helper function for myinfo. Tokenizes my text''' 
    # for each token in the text (the result of text.split(),
    # apply a function that strips punctuation and converts to lower case.
	tokens = map(lambda x: x.strip(',.&').lower(), text.split())
	tokens = list(filter(None, tokens))
	return tokens

def myinfo(line): 
	'''Extracts all info from json file'''
	temp = json.loads(line)
	stars = int(temp['stars'])
	text = temp['text']
	mytoken = tokenize(text)
	return stars, text, mytoken 

def remove_stopwords(mydata): 
	'''Removes stopwords from data frame''' 
	stemmer = PorterStemmer()
	words = stopwords.words("english")
	mydata['cleaned'] = mydata['text'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z]", " ", x).split() if i not in words]).lower())
	return mydata

def bigrams(text):
    # our unigrams are our tokens
    unigrams=tokenize(text)
    # the bigrams just contatenate 2 adjacent tokens with _ in between
    bigrams=list(map(lambda x: '_'.join(x), zip(unigrams, unigrams[1:])))
    bigrams = str(bigrams)[1:-1] 
    # returning a list containing all 1 and 2-grams
    return bigrams

def uni_bi(text): 
    unigrams=tokenize(text)
    unigrams = str(unigrams)[1:-1]
    # the bigrams just contatenate 2 adjacent tokens with _ in between
    bigrams=list(map(lambda x: '_'.join(x), zip(unigrams, unigrams[1:])))
    bigrams = str(bigrams)[1:-1] 
    return unigrams+bigrams

def stringher(example): 
    example = ''.join(example.split(',')[::-1])
    example = example.replace("'", "")
    return example 

def svm_model(feature, k, max_iter):
    vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
    final_features = vectorizer.fit_transform(mydata[feature]).toarray()

    X = mydata[feature]
    Y = mydata['Stars']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
    pipeline = Pipeline([('vect', vectorizer),
    	('chi',  SelectKBest(chi2, k=k)),
    	('clf',  SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=0,max_iter=max_iter, tol=None))])
    model = pipeline.fit(X_train, y_train)
    ytest = np.array(y_test)
    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))
    return model 



if __name__ == '__main__':

	# Read in data and make a data frame 
	path = "yelp/yelp_academic_dataset_review.json"
	lines = open(path, encoding="utf8").readlines()
	information = list(map(myinfo, lines[0:50000])) #500000
	text_list = list(map(itemgetter(1), information)) 
	star_list = list(map(itemgetter(0), information)) 
	token_list = list(map(itemgetter(2), information))
	token_list_big = list(map(lambda x: " ".join(x), token_list))
	data_tuples = list(zip(star_list, token_list_big))
	mydata = pd.DataFrame(data_tuples, columns=['Stars','text'])

	# Remove stopwords 
	mydata = remove_stopwords(mydata)

	# Create unigrams and bigrams 
	mydata['bigram'] = list(map(bigrams, mydata['cleaned']))
	mydata['bigram'] = list(map(lambda x:stringher(x), mydata['bigram']))
	mydata['unibi'] = mydata['bigram'] + mydata['cleaned']

	# Create models 
	mod1 = svm_model(feature = 'cleaned', k=120, max_iter = 50)
	mod2 = svm_model(feature = 'bigram', k = 120, max_iter= 50)
	mod3 = svm_model(feature = 'unibi', k = 120, max_iter= 50)
	mod4 = svm_model(feature = 'cleaned', k=1200, max_iter = 100)
	mod5 = svm_model(feature = 'bigram', k=1200, max_iter = 100)
	mod6 = svm_model(feature = 'unibi',k=1200, max_iter = 100)
	mod7 = svm_model(feature = 'bigram', k=120, max_iter = 100)
	mod8 = svm_model(feature = 'unibi',k=1200, max_iter = 100)

	with open('mod1_svm.pickle', 'wb') as f:
		pickle.dump(mod1, f)

	with open('mod2_svm.pickle', 'wb') as f:
		pickle.dump(mod2, f)

	with open('mod1_svm.pickle', 'wb') as f:
		pickle.dump(mod3, f)


	with open('mod4_svm.pickle', 'wb') as f:
		pickle.dump(mod1, f)

	with open('mod5_svm.pickle', 'wb') as f:
		pickle.dump(mod2, f)

	with open('mod6_svm.pickle', 'wb') as f:
		pickle.dump(mod3, f)

	with open('mod7_svm.pickle', 'wb') as f:
		pickle.dump(mod2, f)

	with open('mod8_svm.pickle', 'wb') as f:
		pickle.dump(mod3, f)

