import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import pandas as pd
import re 
import string 
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec

def preprocess(df):
    '''Tokenizes, removes numbers, punctionation, makes lower case, and remove white space '''
    mylist = []
    nrow = len(df)
    for i in range(0,nrow): 
        myrow = df.iloc[[i]]
        mystring = myrow.to_string(header = False)
        # remove numbers 
        nonum = re.sub(r'\d+', '', mystring)
        # remove punctuation 
        nopunct = "".join([char.lower() for char in nonum if char not in string.punctuation]) 
        # remove white space 
        nowhite = re.sub('\s+', ' ', nopunct).strip()
        # tokenize 
        mytoken = sent_tokenize(nowhite)
        # lower case 
        lower_case = [token.lower() for token in mytoken]
        # as string 
        mystring = ' '.join([str(elem) for elem in lower_case]) 
        # list 
        mylist.append(mystring)
    # Return to text file 
    with open('preprocessed_file.txt', 'w') as filehandle:
        filehandle.writelines("%s\n" % item for item in mylist)
    print("Please check folder for preprocessed text file")
    return(mylist)


def cosine_comparing(mod, model_name): 
	'''Generates a csv of the top 5 similar words'''
	eval_words = ["olympian", "athletes", "winter", "history", "jeopardy", "demon", "child", "marxist", "california", "religion"]
	list_of_tuples = []
	for i in range(0,10):
		top5 = mod.most_similar(eval_words[i])[:5]
		list_of_tuples.append(top5)
	mod_df = pd.DataFrame(list_of_tuples, index = eval_words, columns = ["First", "Second", "Third", "Fourth", "Fifth"])
	mod_df.to_csv(model_name)


def word2vec_models(mylist): 
   	'''Runs 4 different models and outputs csvs comparing the top 5 cosine simirities''' 
   	list_of_lists = []
   	for line in mylist:
   		stripped_line = line.strip()
   		line_list = stripped_line.split()
   		list_of_lists.append(line_list)

   	model1 = gensim.models.Word2Vec(list_of_lists, sg = 1)
   	model1.save("word2vec.model1")
   	cosine_comparing(model1, "mod1_cosine.csv")

   	model2 = gensim.models.Word2Vec(list_of_lists, sg = 0)
   	model2.save("word2vec.model2")
   	cosine_comparing(model2, "mod2_cosine.csv")

   	model3 = gensim.models.Word2Vec(list_of_lists, size = 50, sg = 1)
   	model3.save("word2vec.model3")
   	cosine_comparing(model3, "mod3_cosine.csv")

   	model4 = gensim.models.Word2Vec(list_of_lists, size = 50, sg = 0)
   	model4.save("word2vec.model4")
   	cosine_comapring(model4, "mod4_cosine.csv")

   	print("Please check folder for CSVs")




if __name__ == '__main__':
	print("Lab Work")
	jeopardy = pd.read_json("JEOPARDY_QUESTIONS1.json")
	jeopardy_preprocess = preprocess(jeopardy)
	print("Homework Question")
	word2vec_models(mylist = jeopardy_preprocess)
