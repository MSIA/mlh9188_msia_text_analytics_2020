import tika 
import timeit
import matplotlib.pyplot as plt
import nltk 
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
import spacy 
import stanza
import multiprocessing as mp
from tika import parser
import re 


def comparing_nltk_and_spacy(): 
	# Load florida man 
	tika.initVM()
	florida = parser.from_file('floridaman.pdf')
	florida_content = florida["content"]
	print("Article is loaded")

	# NLTK work 
	# Tokenization 
	nltk_sents = sent_tokenize(florida_content)
	nltk_words = word_tokenize(florida_content)

	nltk_sents_time = timeit.timeit('lambda: word_sents(florida_conent)')
	nltk_words_time = timeit.timeit('lambda: sent_tokenize(florida_conent)')
	print("NLTK tokens")
	print("NLTK Sample of sents:")
	print(nltk_sents[0:4])
	print("NLTK Sample of words:")
	print(nltk_words[0:4])


	# Stemming 
	ps = PorterStemmer() 
	def nltk_stem(): 
	    for w in nltk_words: 
	        print(w, " : ", ps.stem(w))         
	nltk_stem_time = timeit.timeit('lamda: nltk_stem')

	mylist = []
	for w in nltk_words: 
	    myvalue = (w, ":", ps.stem(w))
	    mylist.append(myvalue)
	print("NLTK Sample of stemming:")
	print(mylist[0:4])

	# POS tagging 
	nltk_pos = nltk.pos_tag(nltk_words)
	nltk_pos_time = timeit.timeit('lambda: nltk.pos_tag(nltk_words)')
	print("NLTK Sample of pos tagging:")
	print(nltk_pos[0:4])

	
	# Spacy work 
	npl = spacy.load('en_core_web_sm')
	npl.add_pipe(npl.create_pipe('sentencizer'))

	# Spacy tokens 
	florida_content_spacy = npl(florida_content)
	spacy_sents = [sent.string.strip() for sent in florida_content_spacy.sents]
	spacy_words = [token.text for token in florida_content_spacy]
	print("Spacy tokens")
	print("Spacy Sample of sents:")
	print(spacy_sents[0:4])
	print("Spacy Sample of words:")
	print(spacy_words[0:4])
	spacy_sents_time = timeit.timeit('lambda: [sent.string.strip() for sent in florida_content_spacy.sents]')
	spacy_words_time = timeit.timeit('lambda: [token.text for token in florida_content_spacy]')

	# Spacy lemmatization 
	def spacy_lemma():
	    for token in florida_content_spacy: 
	        print(token.text, token.lemma_)
        
	spacy_lemma_time = timeit.timeit('lambda: spacy_lemma')

	mylist2 = [] 
	for token in florida_content_spacy: 
	    myvalue = (token.text, token.lemma)
	    mylist2.append(myvalue)

	print("Spacy Sample of lemmatization:")
	print(mylist2[0:4])


	# Spacy pos 
	def spacy_pos():
	    for token in florida_content_spacy:
	        print(token.text, token.pos_, token.tag_)
	        
	spacy_pos_time = timeit.timeit('lambda: spacy_pos')

	mylist3 = [] 
	for token in florida_content_spacy:
		myvalue = (token.text, token.pos_, token.tag_)
		mylist3.append(myvalue)

	print(mylist3[0:4])

	# Plots 
	print("Please see files to find plot comparisons")
	packages = ['NLTK', "Spacy"]

	# tokenization -sents 
	def sent_plot():
		mytime = [nltk_sents_time, spacy_sents_time]
		plt.bar(packages,mytime)
		plt.title('Comparing tokenization timing: sents')
		plt.savefig('sents.png', dpi=300, bbox_inches='tight')
		plt.close()
	sent_plot()

	# tokenization - words 
	def word_plot():
		mytime2 = [nltk_words_time, spacy_words_time]
		plt.bar(packages,mytime2)
		plt.title('Comparing tokenization timing: words')
		# save the figure
		plt.savefig('words.png', dpi=300, bbox_inches='tight')
		plt.close()
	word_plot()


	# lemmatization/stemming
	def lemm_sent_plot():
		mytime3 = [nltk_stem_time, spacy_lemma_time]
		plt.bar(packages,mytime3)
		plt.title('Comparing timing: stemming/lemmatization')
		# save the figure
		plt.savefig('stem_lemma.png', dpi=300, bbox_inches='tight')
		plt.close()
	lemm_sent_plot()

	# pos tagging 
	def pos_plot():
		mytime4 = [nltk_pos_time, spacy_pos_time]
		plt.bar(packages,mytime4)
		plt.title('Comparing timing: pos')
		# save the figure
		plt.savefig('pos.png', dpi=300, bbox_inches='tight')
		plt.close()
	pos_plot()



def find_emails(): 
	# load maternity 
	tika.initVM()
	maternity = parser.from_file('maternity.pdf')
	maternity_content = maternity["content"]
	print("Maternity file loaded")

	# Find all email addresses 
	maternity_sents = sent_tokenize(maternity_content)
	maternity_words = word_tokenize(maternity_content)

	num_sents_m = len(maternity_sents)
	for i in range(0,num_sents_m):
	    line = maternity_sents[i]
	    match = re.search(r'[\w\.-]+@[\w\.-]+.', line)  
	    if match != None:
	    	print(match)

	# Load florida 
	florida = parser.from_file('floridaman.pdf')
	florida_content = florida["content"]
	print("Florida rticle is loaded")

	florida_sents = sent_tokenize(florida_content)
	florida_words = word_tokenize(florida_content)

	num_sents_f = len(florida_sents)
	for i in range(0,num_sents_f):
		line = florida_sents[i]
		match = re.search(r'[\w\.-]+@[\w\.-]+.', line)  
		if match != None:
			print(match)


	# My test 
	mytest = "This is an email: megan@aol.com. This is not an email megan@. This is also an email hello@hello.org"
	mytest_sents = sent_tokenize(mytest)

	for i in range(0, len(mytest_sents)): 
		line = mytest_sents[i]
		match = re.search(r'[\w\.-]+@[\w\.-]+.', line) 
		if match != None: 
			print(match)




def get_dates(sents):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    num_sents = len(sents)
    for i in range(0,num_sents):
        line = sents[i]
        match = re.search(r'\d{1,2}/\d{1,2}/\d{4}', line)
        match2 = re.search(r'\d{1,2}-\d{1,2}-\d{4}', line)
        match3 = re.search(r'\d{4}/\d{1,2}/\d{1,2}', line)
        match4 = re.search(r'\d{4}-\d{1,2}-\d{1,2}', line)
        if match != None:
            print(match)
        if match2 != None: 
            print(match2)
        if match3 != None: 
            print(match3)
        if match4 != None:
            print(match4)
        month_key = any(months in line for months in months)
        if month_key != False: 
            match5 = re.search(r'\b[A-Z].*?\b[" "]{1}\d{1,2}[\,\.]{1}[" "]{1}\d{4}', line)  
            if match5 != None: 
                print(match5)



if __name__ == '__main__':
	print("Question 1 work ")
	comparing_nltk_and_spacy()
	print("Question 2a work- emails")
	find_emails()
	print("Question 2b work- dates")
	tika.initVM()
	florida = parser.from_file('floridaman.pdf')
	florida_content = florida["content"]
	florida_sents = sent_tokenize(florida_content)
	maternity = parser.from_file('maternity.pdf')
	maternity_content = maternity["content"]
	maternity_sents = sent_tokenize(maternity_content)
	date_line = sent_tokenize("This is a date January 11, 2020. This is not a date 444/11/2222. This is a date 12/22/2030.")

	get_dates(florida_sents)
	get_dates(maternity_sents)

