import pandas as pd
import math
import numpy as np
import sys
import os

try:
   import cPickle as pickle
except:
    import pickle

def save_dict(dic, path):
    with open(path, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b

test_path = sys.argv[1]
output_path = sys.argv[2]

tag_list = [
        "SCIENCE-FICTION",
        "SPECULATIVE-FICTION",
        "FICTION",
        "NOVEL",
        "FANTASY",
        "CHILDREN'S-LITERATURE",
        "HUMOUR",
        "SATIRE",
        "HISTORICAL-FICTION",
        "HISTORY",
        "MYSTERY",
        "SUSPENSE",
        "ADVENTURE-NOVEL",
        "SPY-FICTION",
        "AUTOBIOGRAPHY", "HORROR",
        "THRILLER",
        "ROMANCE-NOVEL",
        "COMEDY",
        "NOVELLA",
        "WAR-NOVEL",
        "DYSTOPIA",
        "COMIC-NOVEL",
        "DETECTIVE-FICTION",
        "HISTORICAL-NOVEL",
        "BIOGRAPHY",
        "MEMOIR",
        "NON-FICTION",
        "CRIME-FICTION",
        "AUTOBIOGRAPHICAL-NOVEL",
        "ALTERNATE-HISTORY",
        "TECHNO-THRILLER",
        "UTOPIAN-AND-DYSTOPIAN-FICTION",
        "YOUNG-ADULT-LITERATURE",
        "SHORT-STORY",
        "GOTHIC-FICTION",
        "APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION",
        "HIGH-FANTASY"]

def read_data(path):
    print ('Reading data from ',path)
    with open(path,'r') as f:

        tags = []
        articles = []
        tags_list = []

        f.readline()
        for line in f:
            start = line.find(',')
            article = line[start+1:]

            articles.append(article)

    return articles

mapping = load_dict('answer_dict')
X_test = read_data(test_path)

with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)

    for index in range(1234):
        print(X_test[index])
        labels = mapping[ X_test[index] ]
        assert len(labels)>0
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
