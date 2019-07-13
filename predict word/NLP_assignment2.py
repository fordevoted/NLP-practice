from nltk.collocations import BigramCollocationFinder
import re
import codecs
import numpy as np
import string

#print("enter train1")
def train_language(path,lang_name):
    print("enter train")
    words_all = []
    translate_table = dict((ord(char),None)for char in string.punctuation)
    with codecs.open(path,"r","utf-8") as filep:
        for i , line in enumerate(filep):
           # print("before",line)
            line = " ".join(line.split()[0:])
            
            line = line.lower()
            line = re.sub(r"\d+","",line)
            if len(line)!= 0:
                line = line.translate(translate_table)
                line = re.sub(' +',' ',line)
                words_all+=(line.split())
                
                #words_all.append("\n")
    
    #all_str = ''.join(words_all)
    #print("seq_all is ",all_str)
   # all_str = re.sub(' +',' ',all_str)
    #print("seq_all is ",all_str)
    seq_all =[i for  i  in words_all]
    
    finder = BigramCollocationFinder.from_words(seq_all)
   # print("after",finder)
    #finder.apply_freq_filter(2)
    bigram_model = finder.ngram_fd
    bigram_model = bigram_model.items()
    bigram_model =  sorted(finder.ngram_fd.items(), key=lambda item: item[1],reverse=True)

    print( bigram_model)
    np.save(lang_name+".npy",bigram_model)

if True:
    root="train\\"
    lang_name = ["french","english","german","italian","dutch","spanish"]
    train_lang_path = ["dataset.txt"]
    for i , p in enumerate(train_lang_path):
        print("enter train")
        train_language(root+p,lang_name[i])