from nltk.corpus import stopwords
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS
from nltk.probability import FreqDist as FD
import string
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pnds
import matplotlib.pyplot as mplp
import nltk
import operator
nltk.download('punkt')

#Importing Files

f1 = open("Cryptography_by_Cristopher.txt", "r", encoding="utf-8")

# Main Content extractor

# For taking the main lines of the book 1 : 'Cryptography_by_Cristopher'

with open("Cryptography_by_Cristopher.txt", encoding="utf-8") as book:
    lines_1 = book.readlines()

begin_index = lines_1.index("Chapter 1\n")
end_index = len(lines_1) - 1 - lines_1[::-1].index("Certificates\n")

print("The main content of the book for text T1 is from line numbers {} to {}".format(
   begin_index, end_index))

lines_1 = lines_1[begin_index:end_index]

# ss3 part 

def empty_line_remover(lines):  # function for removing the running words & empty lines

    chapter_pattern = r"Chapter [IVX]+"

    temp = []
    for line in lines:
        is_valid = ((line == '\n') or re.match(chapter_pattern, line))
        # If the line is neither a chapter number nor a part heading nor an empty line
        if(not is_valid):
            temp.append(line)           # include it in the final list

    return temp


lines_1 = empty_line_remover(lines_1)


T1 = ''.join(lines_1)  # joining the lines to string



T1_words = T1.split()
print("number of words in T1:", len(T1_words))


def text_preprocessor(text):  # Function to clean books

    text = text.lower()
    text = re.sub('https?://\S+|www\.\S+', '', text)  # to remove links
    text = re.sub('\s', '', text)            # Replacing spaces with ''
    # Removing non-alphanumeric characters
    text = re.sub(r'\W+', '', text)
    text = re.sub('', ' ', text)             # Replacing spaces with ''
    return text

T1 = text_preprocessor(T1)
print(T1)  # printing the cleaned text of book T1

token1 = word_tokenize(T1)  # Tokenizing the book T1
print(token1)

fd1 = FD(token1)  # frequency distribution of token 1
print(fd1)

# Frequency distribution

K = 25
list_fd1 = dict(sorted(fd1.items(), key=operator.itemgetter(1), reverse=True))
list_output_fd1 = dict(list(list_fd1.items())[0: K])



print("K highest frequency words for text T1 are : " + str(list_output_fd1))


# bar graph of word-frequency of fd1
mplp.bar(list_output_fd1.keys(), list_output_fd1.values())
mplp.xlabel('Words')
mplp.ylabel('Frequency')
mplp.show()

# Word Cloud Formation

# wordcloud of T1
word_cloud_instance = WordCloud(width=800, height=800, background_color='black',
                                #min_font_size=8).generate(T1)

mplp.figure(figsize=(8, 8), facecolor=None)
mplp.imshow(word_cloud_instance)
mplp.axis("off")
mplp.tight_layout(pad=0)
mplp.show()

#StopWords Removal and Cloud Forming

nltk.download('stopwords')

all_stopwords = stopwords.words('english')

# wordcloud of T1 after removing stop word
word_cloud_instance = WordCloud(width=800, height=800, background_color='black',
                                stopwords=all_stopwords, min_font_size=8).generate(T1)

mplp.figure(figsize=(8, 8), facecolor=None)
mplp.imshow(word_cloud_instance)
mplp.axis("off")
mplp.tight_layout(pad=0)
mplp.show()


words = {}


def word_counter(text):  # function for matching number of words to its length

    for word in text.split():

        if(len(word) not in words):
            words[len(word)] = 1
        else:
            words[len(word)] += 1


word_counter(T1)

list_count_t1 = sorted(words.items())
x1, y1 = zip(*list_count_t1)
mplp.plot(x1, y1)
mplp.xticks(range(0, 25))
mplp.rcParams["figure.figsize"] = (15, 5)
mplp.xlabel("Wordlength")
mplp.ylabel("Frequency")
mplp.show()

# PoS Tagging

nltk.download('averaged_perceptron_tagger')

tagged1 = nltk.pos_tag(token1)  # doing pos tagging of token1
tagged1
print(tagged1,"\n")

#bar graph
dict1 = {}
for a, b in tagged1:
    if(b not in dict1):
        dict1[b] = 1
    else:
        dict1[b] += 1

sorted_d1 = dict(
    sorted(dict1.items(), key=operator.itemgetter(1), reverse=True))

N = 20
out1 = dict(list(sorted_d1.items())[0: N])

mplp.bar(out1.keys(), out1.values())
mplp.xlabel('TAGS')
mplp.ylabel('Count')
mplp.show()


# After the Pos Tagging we need to find the  verbs and nouns from the tagged text 

T1_nouns = []
T1_verbs = []

for i in tagged1:
  if(i[1] == "NN" or i[1] == "NNS" or i[1] == "NNP" or i[1] =="NNPS"):
    T1_nouns.append(i)
  elif (i[1]=="VB" or i[1]=="VBD" or i[1]=="VBG" or i[1]=="VBN" or i[1]=="VBP" or i[1]=="VBZ"):
    T1_verbs.append(i)

#print(T1_nouns)
#print(T1_verbs)

# importing wordnet from nltk corpus 
from nltk.corpus import wordnet
nltk.download('wordnet')

# Categorizing Nouns (25 Categories)
T1_nouns_dict = {}

for i in T1_nouns:
  syn = wordnet.synsets(i[0])
  for j in syn:
    if j.lexname()[0]=='n':
      if j.lexname() in T1_nouns_dict:
        T1_nouns_dict[j.lexname()]+=1
      else:
        T1_nouns_dict[j.lexname()]=1

print(T1_nouns_dict)

# Now Categorizing the Verbs (16 Categories)
T1_verbs_dict = {}
for i in T1_verbs:
  syn = wordnet.synsets(i[0])
  for j in syn:
    if j.lexname()[0]=='v':
      if j.lexname() in T1_verbs_dict:
        T1_verbs_dict[j.lexname()]+=1
      else:
        T1_verbs_dict[j.lexname()]=1

print(T1_verbs_dict)

# Histogram for Nouns 

mplp.bar(T1_nouns_dict.keys(), T1_nouns_dict.values(),color='b')
y_pos = range(len(T1_nouns_dict.keys()))
mplp.xticks(y_pos, T1_nouns_dict.keys(), rotation=90)
mplp.show()
   

# Histogram for Verbs 

mplp.bar(T1_verbs_dict.keys(), T1_verbs_dict.values(),color='b')
y_pos = range(len(T1_verbs_dict.keys()))
mplp.xticks(y_pos, T1_verbs_dict.keys(), rotation=90)
mplp.show()


# Second Part  Round 2
# Named Entity Recognition (NER) 
# import spacy , displacy and en_core_web_sm for the same 
import spacy
from spacy import displacy
import en_core_web_sm
from collections import Counter
NER = en_core_web_sm.load()

print([(X.text, X.label_) for X in doc1.ents])

# Extracting text for Manual Labelling for T1

para_T1 = T1[0:5000]
doc1 = NER(para_T1)
displacy.render(doc1, jupyter=True, style='ent')

comp_labels_1 = [x.labels_ for x in doc1.ents]
print(comp_labels_1)

# Manual labelled and performance measures for the text

from sklearn import metrics
manual_labels_1 =['ORDINAL','NULL','NULL','CARDINAL','PERSON','DATE','ORDINAL','NULL','CARDINAL','NULL','CARDINAL','WORK_OF_ART','QUANTITY','CARDINAL','DATE','NULL']

print(metrics.classification_report(manual_labels_1,comp_labels_1))

# Third Part Round 2
