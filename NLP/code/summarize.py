
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
import cPickle as cp
import matplotlib.pyplot as plt
from colorize import *

import os, os.path as osp, time
import codecs
import networkx as nx
from xml.dom.minidom import parseString

import nltk

import tsne
import gensim ## use gensim to load the google news w2vec model

import scipy.spatial.distance as ssd
import scipy.optimize as sopt
import Pycluster as clst

w2v = None
def load_gnews_w2v(fpath='../data/GNews.bin.gz'):
	print(colorize("Loading the Google news word2vec model.", "blue", bold=True))
	print(colorize(" >> This can take up to 5 minutes...", "red", bold=True))
	st = time.time()
	model = gensim.models.Word2Vec.load_word2vec_format(fpath,binary=True)
	print(colorize(" >> DONE loading model [%d s]."%(time.time()-st), "green", bold=True))
	global w2v
	w2v = model

## filter words based on parts of speech.
def pos_filter(tagged, tags=['NN', 'JJ', 'NNP']):
	return [item[0] for item in tagged if item[1] in tags]

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
	pos = nx.spring_layout(G,iterations=1000)
	#nx.draw_networkx_nodes(G,pos,hold=True)
	nx.draw(G,pos,labels=node_labels,node_size=np.exp(2*scores+1)+200,with_labels=True)
	plt.show()


def test_textrank(n=5):
	words = ['ash','bittoo','casasa','d','e']
	W = np.random.rand(n,n)
	W = W - np.diag(np.diag(W))
	
	x1z = textrank_loopy(W,S=np.zeros((n,1)))
	x1o = textrank_loopy(W,S=np.ones((n,1)))
	
	x2 = textrank_mathy(W)
	visualize(words,x2, W)
	
	plt.subplot(3,1,1)
	plt.stem(x1z)
	plt.subplot(3,1,2)
	plt.stem(x1o)
	plt.subplot(3,1,3)
	plt.stem(x2)
	plt.show()


def read_xml_sentences(fname, min_nwrds=3):
	"""s
	Parse the xml DUC02 data file and return 
	a list of sentences with more than min_nmwords.
	"""
	f = open(fname,'r')
	dat = f.read()
	f.close()
	dom = parseString(dat)
	sents =  dom.getElementsByTagName('TEXT')[0].childNodes

	filt_sents = []
	for s in sents:
		if s.attributes:
			if s.attributes.has_key('wdcount'):
				nwrds = int(s.attributes.get('wdcount').value)
				if nwrds > min_nwrds:
					filt_sents.append(str(s.firstChild.data))
	return filt_sents

def in_w2v(wrd):
	"""
	Checks if wrd in word2vec database.
	"""
	try:
		w2v[wrd]
		return True
	except:
		return False

def save_cpickle(words,x, fpath="dvec.cp"):
	fout = open(fpath, 'w')
	cp.dump([words,x],fout)
	fout.close()

def load_cpickle(fpath="dvec.cp"):
	fin = open(fpath,'r')
	words,x = cp.load(fin)
	fin.close()
	return words,x

def get_distance_matrix(words,load_cp=False):
	## load the word2vec database:
	if load_cp:
		return load_cpickle(fpath="dmat.cp")
	else:
		if w2v==None:
			load_gnews_w2v()
			#pass
		w2v_words = [wrd for wrd in words if in_w2v(wrd)]
		N = len(w2v_words)
		W = np.zeros((N,N))
		for i in xrange(N):
			for j in xrange(i+1,N):
				dij = w2v.similarity(w2v_words[i],w2v_words[j])
				W[i,j] = dij
				W[j,i] = dij
		save_cpickle(w2v_words,W,fpath="dmat.cp")
		return w2v_words,W

def get_word_vecs(words,load_cp):
	"""
	return the word-vectors.
	"""
	if load_cp:
		return load_cpickle(fpath="dvec.cp")
	else:
		## load the word2vec database:
		if w2v==None:
			load_gnews_w2v()
		w2v_words = [wrd for wrd in words if in_w2v(wrd)]
		vecs = [w2v[wrd].astype('float') for wrd in w2v_words]
		vecs = np.array(vecs)
		save_cpickle(w2v_words,vecs,fpath="dvec.cp")
		return w2v_words,vecs

def repeat_sample(weights, words):
	"""
	Weights : a vector of length N of relative importance.
	Returns a vector of length N, with the number of 
	integer instances for that element.
	"""
	assert np.min(weights) > 0.0
	assert len(words)==len(weights)

	N = len(words)
	counts = np.ceil(weights / np.min(weights))
	samples = []
	for i in xrange(N):
		for _ in xrange(counts[i]):
			samples.append(words[i])
	return samples

def cluster_words(words, score_set, load_cp, k=3, viz=False):
	fwords,vecs = get_word_vecs(words,load_cp)
	assert set(words)==set(fwords)
	n = len(fwords)
	vec_2d = tsne.bh_sne(vecs, d=2,  perplexity=2.0)
	k_guess = min(n/5,15)
	clusters,_,_ = clst.kcluster(vec_2d,nclusters=8,npass=40)
	clusters = np.array(clusters)
	categories = np.unique(clusters)

	extractive_clusters = []
	abstractive_clusters = []
	for c in categories:
		members = np.arange(n)[clusters==c]
		wrd_weights = [score_set[fwords[i]] for i in members]

		weight_sort = np.argsort(-wrd_weights)
		cweights = wrd_weights[weight_sort] 

		sorted_members = members[weight_sort]
		cwords = [fwords[i] for i in sorted_members]

		w2v_wrds = w2v.most_similar(positive=repeat_sample(cweights,cwor), topn=2)
		abstractive_words = [it[0] for it in w2v_wrds]

		extractive_clusters.append(cwords)
		abstractive_clusters.append(abstractive_words)
	
	if viz:
		plt.figure()
		plt.scatter(vec_2d[:,0], vec_2d[:,1])
		ax = plt.gca()
		for i in xrange(len(filt_words)):
			ax.text(vec_2d[i,0], vec_2d[i,1], filt_words[i])
		plt.show()

	return extractive_clusters, abstractive_clusters

def co_occurence(text, load_cp, ntextrank=4, viz=False):
	"""
	Returns a set of POS filtered words and their
	co-occurence matrix.

	The POS only allows NN, JJ and NNP tags.

	NxN co-occurence matrix of words
	in WRDS (size(WRDS) = N).
	Where, W_ij = min_pos_i,pos_j abs(pos_i-pos_j)

	Sents is a list of sentences.
	"""
	words = []
	sents = []
	wnl = nltk.stem.WordNetLemmatizer()
	good_tags = ['NN', 'NNS', 'JJ', 'NNP', 'NNPS']
	for s in text:
		## tokenize:
		tokens = nltk.word_tokenize(s)
		tagged = nltk.pos_tag(tokens)
		pos_filt = [str(wnl.lemmatize(itm[0])) if itm[1] in good_tags else '@' for itm in tagged]
		sents += pos_filt

	unique_words = sorted(list(set(sents) - set(['@'])))

	## compute word-occurence:
	occur = []
	for wrd in unique_words:
		is_wrd = [x==wrd for x in sents]
		occur.append(np.where(is_wrd))

	## compute co-occurence:
	nuniq = len(unique_words)
	coccur = np.zeros((nuniq,nuniq))
	for i in xrange(nuniq):
		i_loc = np.atleast_2d(occur[i]).T
		for j in xrange(i+1,nuniq):
			j_loc = np.atleast_2d(occur[j]).T
			coccur[i,j] = coccur[j,i] = np.min(ssd.cdist(i_loc,j_loc))

	## get the word2vec distances:
	filt_words,W = get_distance_matrix(unique_words, load_cp)

	## filter out the words which are not in word2vec:
	filt_inds = np.atleast_2d(np.where([w in filt_words for w in unique_words]))
	W_coccur = coccur[filt_inds.T,filt_inds]

	## reciprocate the co-occurence matrix:
	Winv_coccur = np.zeros_like(W_coccur)
	Winv_coccur[np.nonzero(W_coccur)] = 1/ W_coccur[np.nonzero(W_coccur)]

	## only have an edge if the co-occurence is less than NTEXTRANK:
	W_txtrank = W_coccur <= ntextrank
	W_txtrank -= np.diag(np.diag(W_txtrank))

	## combine Word2Vec and Co-occurence:
	W_comb = np.minimum(W,Winv_coccur)
	
	wscores_w2v  = textrank_loopy(W, d=0.85, etol=1e-4)
	wscores_comb = textrank_loopy(W_comb, d=0.85, etol=1e-4)
	wscores_coccur = textrank_loopy(W_coccur, d=0.85, etol=1e-4)
	wscores_txtrank = textrank_loopy(W_txtrank, d=0.85, etol=1e-4)

	isort = np.argsort(-wscores_comb[:,0])
	for i in xrange(len(isort)):
		print filt_words[isort[i]], "\t:", wscores_comb[isort[i],0]

	if viz:
		visualize(filt_words,wscores_txtrank,W_txtrank)

	weights = { 'coccur'  : wscores_coccur,
			  	'w2v'     : wscores_w2v,
			  	'thresh'  : wscores_txtrank,
			  	'comb'    : wscores_comb }

	return filt_words, weights


def get_keywords(text):
	"""
	Gets keywords :
	---------------

		1. Using text-rank under different weights
		   for the edges:
		   		(a) coccur : edge-weights = 1/[co-occurence distance]
		   		(b) w2v    : edge-weights = word2vec cosine similarity
		   		(c) thresh : edge-weights = 1 if co-occurence distance < 4 else 0
		   		(d) comb   : edge-weights = min(w2v_ij, coccur_ij)

		2. K-means clustering on word2vec word-vectors:
		   	[X] Clusters are filtered for stability/ shape
		   	- The "Top 3" are based on scores from text-rank

				(a) Extractive  : Top 3 words per cluster
				(b) Abstractive : Top 3 closest words in w2vec space
				                  to each cluster
	"""


def load_text(fname='AP880314-0110.S'):
	fname = "AP891207-0158.S"
	dpath = '../data/sum_data/doc_tag'
	text = read_xml_sentences(osp.join(dpath,fname))
	w1 = co_occurence(text)




load_text()
#w2v = get_distance_matrix([])

"""
Build in co-occurence and word2vec weights.

Visualize tightness of clusters

Incorporate tf-idf : if out of vocab -- then word is probably
important => high-importance.

How can w2v and textrank be combined?

Lemmatization
http://nbviewer.ipython.org/github/charlieg/A-Smattering-of-NLP-in-Python/blob/master/A%20Smattering%20of%20NLP%20in%20Python.ipynb
"""

