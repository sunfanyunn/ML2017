# coding: utf-8
import re
import numpy as np
from sklearn.manifold import TSNE
import word2vec
from matplotlib import pyplot as plt
from adjustText import adjust_text
import nltk
'''
word2vec.word2phrase('all.txt', 'phrases.txt', verbose=True)
word2vec.word2vec('phrases.txt', 'text.bin', size=100, verbose=True)
word2vec.word2clusters('all.txt', 'clusters.txt', 100, verbose=True)
'''
model = word2vec.load('model.bin')
words = [word for word in model.vocab[:500]]
X = [ model[word] for word in words]
X = np.array(X)
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)


def plot_scatter(x,y,texts,adjust=False):

    fig, ax = plt.subplots()
    ax.plot(x, y, 'bo')

    texts = [plt.text(x[i], y[i], texts[i]) for i in range(len(x))]
    if adjust:
        plt.title(str( adjust_text(texts, x, y, arrowprops=dict(arrowstyle='->', color='red')))+' iterations')
    plt.savefig("500")

def func1():
#    pattern = re.compile(r"[,.:;!?’]")
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]
    X, Y, texts = [], [], []
    for i,word in enumerate(words):
#        if not pattern.findall(word):
        if all(c not in word for c in puncts):
            tag = nltk.pos_tag([word])
            if tag[0][1] != 'JJ' and tag[0][1] != 'NNP' and tag[0][1] != 'NN' and tag[0][1] != 'NNS':
                continue
            X.append(X_tsne[i][0])
            Y.append(X_tsne[i][1])
            texts.append(word)

    print(len(X))
    plot_scatter(X, Y, texts, True)

def func2():

    vocabs = words
    reduced = X_tsne
    # filtering
    use_tags = set(['JJ', 'NNP', 'NN', 'NNS'])
    puncts = ["'", '.', ':', ";", ',', "?", "!", u"’"]


    plt.figure()
    texts = []
    cnt = 0
    for i, label in enumerate(vocabs):
        pos = nltk.pos_tag([label])
        if ( len(label) > 1 and pos[0][1] in use_tags
                and all(c not in label for c in puncts)):

            x, y = reduced[i, :]
            cnt+=1
            texts.append(plt.text(x, y, label))
            plt.scatter(x, y)
    print(cnt)
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='k', lw=0.5))

    plt.savefig('hp.png', dpi=600)
#    plt.show()

func2()
