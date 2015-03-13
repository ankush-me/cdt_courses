
## Good Script:
## https://github.com/davidadamojr/TextRank/blob/master/textrank.py
#
# http://networkx.github.io/documentation/latest/examples/drawing/knuth_miles.html

"""
We need to find "important" sentences.

We have vector representations of sentences.
Which means, these sentences lie in some n-dimensional space.
Think of them as points on a plane.

We can definitely cluster the points
to find major trends.

Kind of need to find "outliers"/ statistical anomalies.

"""
import numpy as np 
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssd
import tsne
import nltk
import os, os.path as osp, time
import gensim ## use gensim to load the google news w2vec model
from colorize import *
import codecs
import scipy.optimize as sopt
import networkx as nx
from xml.dom.minidom import parseString


def load_gnews_w2v(fpath='../data/GNews.bin.gz'):
	print(colorize("Loading the Google news word2vec model.", "blue", bold=True))
	print(colorize(" >> This can take up to 5 minutes...", "red", bold=True))
	st = time.time()
	model = gensim.models.Word2Vec.load_word2vec_format(fpath,binary=True)
	print(colorize(" >> DONE loading model [%d s]."%(time.time()-st), "green", bold=True))
	return model

## filter words based on parts of speech.
def pos_filter(tagged, tags=['NN', 'JJ', 'NNP']):
	return [item for item in tagged if item[1] in tags]

def extractKeyphrases(text):
	#tokenize the text using nltk
	wordTokens = nltk.word_tokenize(text)
	pos_tagged = nltk.pos_tag(wordTokens)

	for wrd in pos_filter(pos_tagged):
		print(wrd)
	
	sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
	sentenceTokens = sent_detector.tokenize(text.strip())
	for sent in sentenceTokens:
		print(sent)
		print("---")
	
	#assign POS tags to the words in the text
	tagged = nltk.pos_tag(wordTokens)
	#apply syntactic filters based on POS tags



def textrank_loopy(W, d=0.85, etol=1e-4, S=None):
	## normalize columns of W:
	W = W/(np.spacing(0)+np.sum(W,axis=0))
	[n,_] = W.shape

	if S==None:
		S = np.zeros((n,1))
	err = 1.0
	while err > etol:
		Stmp = (1-d) + d*W.dot(S)
		err = np.linalg.norm(Stmp-S)
		S = Stmp
	return S

def textrank_mathy(W, d=0.85):
	## normalize columns of W:
	W = W/(np.spacing(0)+np.sum(W,axis=0))
	[n,_] = W.shape
	a = (1-d)*np.ones(n)
	sol = sopt.nnls(np.eye(n)-d*W, a)
	print "least-sq residual : ", sol[1]
	return sol[0]

def visualize(words,scores,W):
	N = len(words)
	node_labels = {i:words[i] for i in range(N)}
	G = nx.to_networkx_graph(W)
	pos = nx.spring_layout(G)
	#nx.draw_networkx_nodes(G,pos,hold=True)
	nx.draw(G,pos,labels=node_labels,node_size=np.exp(6*scores+1)+200,with_labels=True)
	plt.show()


def test_textrank(n=5):
	words = ['ash','bittoo','casasa','d','e']
	W = np.random.rand(n,n)
	W = W - np.diag(np.diag(W))
	# W = np.zeros((n,n))
	# W[1:,0] = 1
	# W[0,1:] = 1
	x1z = textrank_loopy(W,S=np.zeros((n,1)))
	x1o = textrank_loopy(W,S=np.ones((n,1)))
	
	x2 = textrank_mathy(W)
	visualize(words,x2, W)
	
	# print (x1.T-x2).shape
	# print np.linalg.norm(x1.T-x2)
	plt.subplot(3,1,1)
	plt.stem(x1z)
	plt.subplot(3,1,2)
	plt.stem(x1o)
	plt.subplot(3,1,3)
	plt.stem(x2)
	plt.show()

def read_sentences(fname, min_nwrds=3):
	f = open(fname,'r')
	dat = f.read()
	f.close()
	dom = parseString(dat)
	sents =  dom.getElementsByTagName('s')
	filt_sents = [s for s in sents if int(s.attributes.get('wdcount').value) > min_nwrds]
	text_sents = [str(s.firstChild.data) for s in filt_sents]
	return text_sents

#model = load_gnews_w2v()
def load_text():
	dpath = '../data/sum_data/doc_tag'
	fname = 'AP880314-0110.S'
	text = read_sentences(osp.join(dpath,fname))
	for t in text:
		print(t+'\n')
	#extractKeyphrases(text)

load_text()


	