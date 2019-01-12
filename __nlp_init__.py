'''
This is the initiation page for NLP. It contains the necessary libraries and some of the sub-routines necessary for NLP.
'''

print('\n', 'Welcome to our Natural Language Processing Library. I am loading at the moment and will be done in a nanosecond...','\n')


import spacy
print('Imported spacy')
import pandas as pd
print('Imported pandas as pd')
import numpy as np
print('Imported numpy as np')
import nltk
print('Imported nltk')
from nltk.tokenize.toktok import ToktokTokenizer
print('Imported ToktokTokenizer')
from nltk.stem import PorterStemmer
print('Imported PortStemmer')
from nltk.tokenize import sent_tokenize, word_tokenize
print('Imported sentence and word tokenizer')
import re
print('Imported re')
from bs4 import BeautifulSoup
print('Imported BeautifulSoup')
from contractions import CONTRACTION_MAP
print('Imported CONTRACTION_MAP')
import unicodedata
print('Imported unicodedata')
import PyPDF2
print('Imported PyPDF2')
from gensim.summarization import summarize
print("Imported gensim's summarize")
import textract
print('Imported textract')
from gensim.summarization import keywords
print("Imported gensim's keywords")
import pyphen
print('Imported pyphen')


# Needed for summarizing and reading in text from PDF
import logging
print("Imported logging for summarizing and reading text from PDFs")
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)
print('*' *80, '\n')



print('DONE! Phew, I am good! Record Time!', '\n\nI also loaded the following functions:\n\n\tpull_all_text(df)\n\tadd_to_library(df)\n\tclean_the_text(raw_text)\n\tdrop_onenglish_words(text)\n\tremove_puntuation(text)\n\tstrip_html_tags(text)\n\tremove_accented_characters(text)\n\texpand_contractions(text, contraction_mapping=CONTRACTION_MAP\n\tremove_special_characters(text, remove_digits=False)\n\tsimple_stemmer(text)\n\tlemmatize_text(text)\n\tremove_stopwords(text, is_lower_case=False)\n\tnormalize_corpus(corpus, html_stripping=True, contraction_expansion=True, accented_char_removal=True, text\n\t\t_lower_case=True, text_lemmatization=True, special_char_removal=True, stopword_removal=True, \n\t\tremove_digits=True)\n\timport_pdf(file_path)\n\ttokenize_by_sentences(text)\n\ttokenize_by_words(text)\n\tfind_keywords(text)\n\tpos_tag(text)\n\tnormalize_corpus(text)')


# nlp = spacy.load('en_core', parse=True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


def pull_all_text(urllibrary):
    import requests
    import pandas as pd
    import time
    
    urllibrary['raw_text'] = 0
    
    #The TRY wrapper is to suppress EXCEPTIONS that may crop up to alert rather than stop the code. 
    try: 
        index = 0
        for url in urllibrary['url']: #looks at each URL in the urllibrary file 
            page = requests.get(url, allow_redirects=False) #pulls the html associated with the URL 
            urllibrary.loc[index,'raw_text'] = page.content #places the raw html into the appropriate row in the dataframe
            urllibrary.loc[index, 'pull_date'] = pd.datetime.today().strftime("%m/%d/%Y")#Time stamps the pull
            print("Processing line", index+1,"of", len(urllibrary['url']), ": text pulled from the", urllibrary.loc[index,'company'],"URL") #This line just lets us know its working

            # This time.sleep() function as a way of putting in a pause so we aren't shut down by ISPs for DOS Attack
            if index%25==0:
                print('\n', '*' *25, '\n',"Pause regulator initiated", '\n', '*' *25)
                time.sleep(3)
                
            index += 1
            
    except Exception as ex:

        print("*" *10, '\n''WARNING: An Exception was thrown at dataset line', index+1,'\n',"*" *10, '\n')
        print(ex)
        nofills = []
        for text in urllibrary['raw_text']:
            if urllibrary.loc[index, 'raw_text'] is None:
                nofills.append(urllibrary.loc[index, 'company'])
                index += 1
            else:
                index += 1
            print(nofills, '\n','*' *25)

        print("*" *10, '\n')
        print('The privacy statements for the following companies were not pulled:')

'''
****************************************************
The add_to_library(urllibrary) function:

This function looks over our urllibrary for [text] columns with NAN values and pulls the text for those URLs.
****************************************************
'''

def add_to_library(urllibrary):
    import requests
    import pandas as pd
    
    index = 0
    for text in urllibrary['raw_text']: #looks at each URL in the urllibrary file
        if text == 0:
            print('index:', index, 'is NaN')
            page = requests.get(urllibrary[index, 'url']) #pulls the html associated with the URL 
            urllibrary.loc[index,'raw_text'] = page.content #places the raw html into the appropriate row in the dataframe
            urllibrary.loc[index, 'pull_date'] = pd.datetime.today().strftime("%m/%d/%Y")#Time stamps the pull
            print("line", index+1,"of", len(urllibrary['url']), "processed")#This line just lets us know its working
            index +=1
        else:
            print("line", index+1,"of", len(urllibrary['url']), "processed")#This line just lets us know its working
            index +=1
            pass
   
        
# Removing HTML Tags
def strip_html_tags(text):
    import bs4 as BeautifulSoup
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def remove_html_tags(text):
    """Remove html tags from a string"""
    import re
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


# Removing Accent characters
def remove_accented_characters(text):
    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# Expanding Contractions
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())                       
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text
      
      
# Removing Special Characters
'''
*********************************************************
This special character removal function uses a variety of techniques. It does not clean 100%, but it gets
us to a place where we can pull keywords and other important information from the text. 

Because of its effectiveness among the functions I have written in this code for cleaning, the clean_the_text
function is my favorite, and it is the one I use below.
*********************************************************
'''

def clean_the_text(text, remove_numbers=False):
    print('\n', '@' *75, '\n', 'CLEANING THE TEXT', '\n\n')
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(text, 'lxml')
    
#     print('PRETTYING UP THE TEXT IN THE CLEANING:  ', '\n\t', soup.prettify())
#     text = soup.text
    
    from pattern.web import URL, plaintext
    text = plaintext(text, keep=[], linebreaks=2, indentation=False)

    import unicodedata
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    import re
    clean = re.compile('<.*?>}{')
    text = re.sub(clean, '', text)
    
    text = text.replace("\'", "'")
    text = text.replace('\\n', ' ')
    text = text.replace('\\xc2\\xae', '')
    text = text.replace('\n',' ')
    text = text.replace('\t','')
    text = text.replace('\s+', '')
    text = text.replace('\r\r\r', '')
    text = text.replace('\\xc2\\xa9 ', '')
    text = text.replace('\\xe2\\x80\\x9c', '')
    text = text.replace('xe2x80x93', ',')
    text = text.replace('xe2x88x92', '')
    text = text.replace('\\x0c', '')
    text = text.replace('\\xe2\\x80\\x9d', '')
    text = text.replace('\\xe2\\x80\\x90', '')
    text = text.replace('\\xe2\\x80\\x9331', '')
    text = text.replace('xe2x80x94', '')
    text = text.replace('\x0c', ' ')
    text = text.replace(']', '] ')
    text = text.replace('\\xe2\\x80\\x99', "'")
    text = text.replace('xe2x80x99', "'")
    text = text.replace('\\xe2\\x80\\x933', '')
    text = text.replace('\\xe2\\x80\\x935', '')
    text = text.replace('\\xef\\x82\\xb7', '')
    text = text.replace('\\', '')
    text = text.replace('xe2x80x99', "")
    text = text.replace('xe2x80x9cwexe2x80x9d', '')
    text = text.replace('xe2x80x93', ', ')
    text = text.replace('xe2x80x9cEUxe2x80x9d', '')
    text = text.replace('xe2x80x9cxe2x80x9d', '')
    text = text.replace('xe2x80x9cAvastxe2x80x9d', '')
    text = text.replace('xc2xa0', '')
    text = text.replace('xe2x80x9cxe2x80x9d', '')
    text = text.replace('xe2x80x9c', '')
    text = text.replace('xe2x80x9d', '')
    text = text.replace('tttttt', ' ')
    text = text.replace('activetttt.', '')    
    text = text.replace('.sdeUptttt..sdeTogglettttreturn', '') 
    text = text.replace('ttif', '')
    text = text.replace('.ttt.', ' ')
    text = text.replace(" t t ", ' ')
    text = text.replace('tttt ', '')
    text = text.replace(' tt ', ' ')
    text = text.replace(' t ', ' ')
    text = text.replace(' t tt t', ' ')
    text = text.replace('ttt', '')
    text = text.replace('ttr', '')
    text = text.replace('.display', '')
    text = text.replace('div class', '')
    text = text.replace('div id', ' ')
    text = text.replace('Pocy', 'Policy')
    text = text.replace('xc2xa0a', ' ')
    text = text.replace(' b ', '')
    text = text.replace('rrrr', '')
    text = text.replace('rtttr', '')
    text = text.replace('    ', ' ')
    text = text.replace('   ', ' ')
    text = text.replace('  ', ' ')
    text = text.replace(' r ', ' ')
    text = text.replace(' tr ', ' ')
    text = text.replace(' rr  r  ', ' ')
    text = text.replace('   tt t t rt ', ' ')
    text = text.replace('r rrr r trr ', ' ')
    text = text.replace(' xe2x80x93 ', ' ')
    text = text.replace(' xe6xa8x82xe9xbdxa1xe6x9cx83  ', ' ')
    text = text.replace(' rrr ', ' ')
    text = text.replace(' rr ', ' ')
    text = text.replace('tr ', '')
    text = text.replace(' r ', '')
    text = text.replace("\'", "")
    text = text.replace(' t* ', ', ')
    

    return text

    print('*' *10, 'DROPPING NON-ENGLISH WORDS FROM THE TEXT', '*' *10)
    from nltk.tokenize import word_tokenize
    token_text_w = word_tokenize(text)
    
    import enchant
    d = enchant.Dict('en_US')
    bad_words = []

    for word in token_text_w:
        if d.check(word) is not True:
            bad_words.append(word)
            
    bad_words = set(bad_words)
    
    for word in token_text_w:
        if word in bad_words:
            text = text.replace(word, '')
            
    #Trial of a new way of cleaning the text
    index = 0
    print('\n\n', '*' *10, len(tokenize_by_sentences(a)), '*' *10,'\n\n')
    for sent in tokenize_by_sentences(a):
        if 'js' in sent or 'css' in sent or 'png' in sent or'woff2' in sent or ' div ' in sent or ' meta "" ' in sent or 'span' in sent:
            a = a.replace(sent, '')
            print('\n', '*' * 25,'\n','CLEANING TOKENIZED SENTENCES OF CODE IN INDEX', index, '*' * 25)
            index += 1

            
    return (text)  
 
#Dropping words and characters not found in the English Dictionary
def drop_nonenglish_words(text):
    print('*' *10, 'DROPPING NON-ENGLISH WORDS FROM THE TEXT', '*' *10)
    from nltk.tokenize import word_tokenize
    token_text_w = word_tokenize(text)
    
    import enchant
    d = enchant.Dict('en_US')
    bad_words = []

    for word in token_text_w:
        if d.check(word) is not True:
            bad_words.append(word)
            
    bad_words = set(bad_words)
    
    for word in token_text_w:
        if word in bad_words:
            text = text.replace(word, '')
            
    return (text)  

# PUNCTUATION REMOVAL
def remove_punctuation(text):
    import string
    table = str.maketrans({key: None for key in string.punctuation})
    return text.translate(table)


# Stemming
def simple_stemmer(text):
    from nltk.stem import PorterStemmer
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text
      
# Lemmatization
def lemmatize_text(text):
    from nltk import nlp
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
      
# Remove Stopwords
def remove_stopwords(text, is_lower_case=False):
    stopword_list = nltk.corpus.stopwords.words('english')
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text
      
# Tokenize by words
def tokenize_by_words(text):
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_w = str(word_tokenize(text))
    return token_text_w

#Tokenize by sentences
def tokenize_by_sentences(text):
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_s = sent_tokenize(text)
    return token_text_s

#Find Keywords
def find_keywords(text):
    from gensim.summarization import keywords
    key_words = keywords(text)
    key_words = key_words.replace('_', '')
    key_words = key_words.replace('\n', ' ')
    return key_words

#Get POS tags
def pos_tag(text):
    import nltk
    from nltk import pos_tag
    from nltk.tokenize import sent_tokenize, word_tokenize
    token_text_w = word_tokenize(text)
    return pos_tag(token_text_w)# Normalize Corpus

# Normalize the Corpus
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True, 
                     text_lemmatization=True, special_char_removal=True, 
                     stopword_removal=True, remove_digits=True):
    
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # strip HTML
        if html_stripping:
            doc = strip_html_tags(doc)
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions    
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text    
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits    
        if special_char_removal:
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)  
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)
            
        normalized_corpus.append(doc)
        
    return normalized_corpus
      
# Importing the text from a PDF given the pathway to the PDF      
def import_pdf(file_path):
    import textract
    text = textract.process(file_path)
    return text


# count the frequency of words in a sentence
def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts

#counts number of words of a particular length (dict = string or frequency distribution
def countwords(dic,length):
    tsum=0
    for i in dic:
        if len(i)==length:
            tsum=tsum+dic[i]
            #print(i,dic[i])
    return tsum

# count average number of words per sentence
def avgwordspersentence(words):
    counter=0
    avg=0
    noofsentences=0
    for i in words:
        if(i!='.'):#and i!=','
            counter=counter+1
        else:
            noofsentences+=1
            avg+=counter            
            counter=0
    avg=avg/noofsentences
    return avg


# count the number of words per sentence

def noofsyllabes(corpus):
    import pyphen
    dic = pyphen.Pyphen(lang='en')
    num=0
    for x in corpus:
        s=dic.inserted(x)
        num=num+s.count('-')+1
    return num
