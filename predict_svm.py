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


if __name__ == '__main__':
	# Clean data 
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


	# Load model 
	best_mod = "mod8_svm.pickle"
	loaded_model = pickle.load(open(best_mod, 'rb'))

	feature = "cleaned"
	vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
	final_features = vectorizer.fit_transform(mydata[feature]).toarray()

	sampled_data = mydata.sample(frac = 0.2)
	X = sampled_data[feature]
	ytest = np.array(sampled_data['Stars'])

	predicted = loaded_model.predict(X)
	# prob_predict = loaded_model.predict_proba(X)
	# star_predictions = pd.DataFrame(prob_predict, columns = ['Prob_of_1_Star', 'Prob_of_2_Star', 'Prob_of_3_Star', 'Prob_of_4_Star', 'Prob_of_5_Star'])

	d = {'Actual':ytest,'Predicted':predicted}
	mod_results = pd.DataFrame(d)

	# mod_results_final = mod_results.join(star_predictions)  
	#Write to json 
	mod_results.to_json("predict_svm_results.json")




