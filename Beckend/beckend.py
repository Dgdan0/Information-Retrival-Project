import concurrent
import sys
from collections import Counter, OrderedDict
import itertools
from itertools import islice, count, groupby

import google
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.corpus import stopwords
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from time import time
from timeit import timeit
from pathlib import Path
import pickle
import numpy as np
from google.cloud import storage
import itertools
import math
from inverted_index_gcp import *
from contextlib import closing
from collections import Counter

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


bucket_name = "wikibucket208"

title_path = 'title_index/postings_gcp_title_index/'
body_path = 'text_index/postings_gcp_text_index/'
NUM_BUCKETS = 150

from collections import defaultdict
def _hash(s):
  return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

def tokens2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def get_pickle(bucket_name, path):
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(path)
    ret = pickle.loads(blob.download_as_string())
    return ret

def load_index(type):
    if type == 'body':
        return get_pickle(bucket_name, 'InvertedIndex/body_index.pkl')
    elif type == 'title':
        return get_pickle(bucket_name, 'InvertedIndex/title_index.pkl')
    else:
        return False

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){,24}""", re.UNICODE)

# Tokenizing text
def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower())]
nltk_stop_words = set(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
all_stopwords = nltk_stop_words.union(corpus_stopwords)
def remove_stopwords(tokens):
    return [token for token in tokens if token not in all_stopwords]
def stem_tokens(tokens, stem_model='porter'):
    if stem_model == 'porter':
        porter = PorterStemmer()
        stemmed_tokens = [porter.stem(token) for token in tokens]
    return stemmed_tokens
    

# Helper functions
def get_posting_locs(type, w):
    prefix_dir = 'InvertedIndex/posting_locs/'
    file_name = '_posting_locs.pickle'
    if type == 'body' or type == 'title':
        prefix_dir += type
    else:
        return False
    
    bucket_id = tokens2bucket_id(w)

    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=prefix_dir):
        if blob.name.endswith(f'{prefix_dir}/{bucket_id}{file_name}'):
            with blob.open("rb") as f:
                pl = pickle.load(f)
                ret_pl = {w: pl[w]}
                if ret_pl[w]:
                    return ret_pl
                else:
                    return None

def get_doc_len(type, bucket_id):
    prefix_dir = 'InvertedIndex/doc_length/'
    file_name = 'doc_len.pickle'
    if type == 'body' or type == 'title':
        prefix_dir += type
    else:
        return False

    client = storage.Client()
    for blob in client.list_blobs(bucket_name, prefix=prefix_dir):
        if blob.name.endswith(f'{prefix_dir}/{bucket_id}{file_name}'):
            with blob.open("rb") as f:
                return pickle.load(f)

def get_docs_lengths(type):
    prefix = 'InvertedIndex/bm25'
    file_name = f'{type}_bm25.pickle'
    
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    for blob in blobs:
        if blob.name.endswith(file_name):
            with blob.open("rb") as f:
                bm25_dict = pickle.load(f)
                N = bm25_dict['N']
                avgdl = bm25_dict['avgdl']
                return (N, avgdl)
    return False

# Query info - getting posting list for query
def query2bucket_id(query, tokenized):
    query_bucket_id = {}
    if not tokenized:
        query = tokenize(query)
    tokens = stem_tokens(query)
    query_tokens = [token for token in tokens if token not in all_stopwords]
    
    for q in query_tokens:
        query_bucket_id[q] = tokens2bucket_id(q)
    return query_bucket_id

def id2info(query_bucket_id, type):
    q_pl = {}
    index_instance = load_index(type)
    
    for w, bucket_id in query_bucket_id.items():
        p_l = get_posting_locs(type, w)
        if p_l:
            index_instance.posting_locs = p_l
            
            if w in p_l:
                posting_list = index_instance.read_a_posting_list('', w, bucket_name)
                q_pl[w] = posting_list
        else:
            q_pl[w] = None
    
    return q_pl

def query_info(query, type, tokenized=False):
    query_bucket_id = query2bucket_id(query, tokenized)
    q_pl = id2info(query_bucket_id, type)
    return q_pl


# Searching function
def calculate_BM25_class(k1, b, query, type, num_res=100):
    N, avgdl = get_docs_lengths(type)
    bm25_scores = {}
    query_pl = query_info(query, type)
    doc_lengths_by_bucket = {}
    
    for term, posting_list in query_pl.items():  # Make better with tokens2bucket_id on doc_id and grouping them together for less reading files
        if posting_list:
            df = len(posting_list)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            bucketed_docs = {}
            
            for doc_id, tf in posting_list:
                bucket_id = tokens2bucket_id(str(doc_id))
                if bucket_id not in bucketed_docs:
                    bucketed_docs[bucket_id] = []
                bucketed_docs[bucket_id].append((doc_id, tf))
            
            for bucket_id, docs in bucketed_docs.items():
                if bucket_id not in doc_lengths_by_bucket:
                    doc_lengths_by_bucket[bucket_id] = get_doc_len(type, bucket_id)
                
                for doc_id, tf in docs:
                    doc_len = doc_lengths_by_bucket[bucket_id].get(doc_id, 0)
                    norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                    score = idf * norm_tf
                    bm25_scores[doc_id] = bm25_scores.get(doc_id, 0) + score
    sorted_bm25_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_bm25_scores[:num_res]


def backend_search(query):
    k1 = 1.5
    b = 0.75
    res_score = {}
    
    title_score = calculate_BM25_class(k1, b, query, 'title')
    body_score = calculate_BM25_class(k1, b, query, 'body')
    
    for doc_id, score in title_score:
        res_score[doc_id] = res_score.get(doc_id, 0) + score * 0.5

    for doc_id, score in body_score:
        res_score[doc_id] = res_score.get(doc_id, 0) + score * 0.5

    # Sort res_score by score in descending order
    sorted_res_score = sorted(res_score.items(), key=lambda x: x[1], reverse=True)

    return [str(doc_id) for doc_id, _ in sorted_res_score]
