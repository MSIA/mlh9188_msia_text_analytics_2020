import tarfile
import pandas as pd 
import json 
import multiprocessing
from collections import Counter
import statistics
from operator import itemgetter 
import tokenize


def tokenize(text):
	tokens = map(lambda x: x.strip(',.&').lower(), text.split())
	tokens = list(filter(None, tokens))
	return tokens

def myinfo(line): 
	temp = json.loads(line)
	stars = int(temp['stars'])
	text = temp['text']
	mytoken = tokenize(text)
	return stars, text, mytoken 


if __name__ == '__main__':
	path = "yelp/yelp_academic_dataset_review.json"
	lines = open(path, encoding="utf8").readlines()

	# Number of documents 
	num_docs = len(lines)
	print("Number of documents is: ", num_docs)

	# Get my info 
	information = list(map(myinfo, lines[0:50000]))

	# Number of labels 
	star_list = list(map(itemgetter(0), information)) 
	labels = set(star_list)
	print("Number of labels are: ", len(labels))
	print("The labels are: ", labels)

	# Distribution of labels 
	star_dist = Counter(star_list)
	print("The distribution of labels is: ", star_dist)

	# Summary statistics around text
	text_list = list(map(itemgetter(1), information)) 
	text_lengths = list(map(len, text_list))
	text_mean = statistics.mean(text_lengths)
	text_min = min(text_lengths)
	text_max = max(text_lengths)
	print("The average review is ", text_mean, " characters.")
	print("The minimum review is ", text_min, " characters.")
	print("The maximum review is ", text_max, " characters.")

	# Tokens 
	token_list = list(map(itemgetter(2), information))
	words = list(map(len, token_list))
	mean_words = statistics.mean(words)
	print("The mean number of words is ", mean_words)

