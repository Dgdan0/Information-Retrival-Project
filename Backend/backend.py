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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import math
import heapq

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()


bucket_name = "ir_project208"

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

def stem_tokens(tokens, stem_model=None):
    if stem_model == 'porter':
        porter = PorterStemmer()
        stemmed_tokens = [porter.stem(token) for token in tokens]
    elif stem_model == 'snowball':
        snowball = SnowballStemmer(language='english')
        stemmed_tokens = [snowball.stem(token) for token in tokens]
    elif stem_model == None:
        return tokens
    else:
        raise ValueError("Unsupported stem model: {}".format(stem_model))
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
    df_thresh = 1000000
    df_thresh_2 = 2000000
    q_pl = {}
    index_instance = load_index(type)
    
    query_toks = list(query_bucket_id.keys())  # Make a list of keys to iterate over
    query_df = {q: index_instance.df.get(q, None) for q in query_bucket_id.keys()}
    
    
        # remove tokens that have 2 million df's
    for tok in query_toks:
        if query_df[tok] != None and query_df[tok] > df_thresh_2:
            del query_bucket_id[tok]
    
    query_toks = list(query_bucket_id.keys())  # Make a list of keys to iterate over
    
    if len(query_bucket_id.keys()) > 3:          # remove tokens that have 1 million df's
        for tok in query_toks:
            if query_df[tok] != None and query_df[tok] >= df_thresh:
                del query_bucket_id[tok]
                    
                
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
def calculate_BM25_score(k1, b, N, avgdl, query_pl, term, type, doc_lengths_by_bucket):
    bm25_scores = []
    posting_list = query_pl[term]
    pl_thresh = 500000
    pl_slice = 100000
    
    if posting_list:        
        if len(posting_list) > pl_thresh:
            posting_list = posting_list[:pl_slice]
        df = len(posting_list)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        bucketed_docs = defaultdict(list)  # Use defaultdict for convenience

        for doc_id, tf in posting_list:
            bucket_id = tokens2bucket_id(str(doc_id))
            bucketed_docs[bucket_id].append((doc_id, tf))

        for bucket_id, docs in bucketed_docs.items():
            doc_lengths = doc_lengths_by_bucket.get(bucket_id)
            if doc_lengths is None:
                doc_lengths = get_doc_len(type, bucket_id)
                doc_lengths_by_bucket[bucket_id] = doc_lengths

            for doc_id, tf in docs:
                doc_len = doc_lengths.get(doc_id, 0)
                norm_tf = tf * (k1 + 1) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                score = idf * norm_tf
                bm25_scores.append((doc_id, score))
    return bm25_scores

def parallel_calculate_BM25_class(k1, b, query, type, num_res=50):
    N, avgdl = get_docs_lengths(type)
    query_pl = query_info(query, type)
    doc_lengths_by_bucket = {}
    bm25_scores = defaultdict(float)
    
    if len(query_pl.keys()) > 1:
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(calculate_BM25_score, k1, b, N, avgdl, query_pl, term, type, doc_lengths_by_bucket) for term in query_pl.keys()]
            for future in as_completed(futures):
                for doc_id, score in future.result():
                    bm25_scores[doc_id] += score

    else:
        if query_pl:
            term = list(query_pl.keys())[0]
            a = calculate_BM25_score(k1, b, N, avgdl, query_pl, term, type, doc_lengths_by_bucket)
            for doc_id, score in a:
                bm25_scores[doc_id] += score
        else:
            return []
        
    return heapq.nlargest(num_res, bm25_scores.items(), key=lambda item: item[1])


def combine_scores(title_scores, body_scores, title_weight = 0.2, body_weight = 0.8, num_res=50):
    combined_scores = defaultdict(float)  # Use float for scores calculation
    
    if title_scores:
        for doc, score in title_scores:
            combined_scores[doc] += score * title_weight

    if body_scores:
        for doc, score in body_scores:
            combined_scores[doc] += score * body_weight
    
    return sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:num_res]

def get_title(doc_ids):
    ret = []
    title_prefix = 'InvertedIndex'
    title_id_dict = {}
    
    client = storage.Client()
    blobs = client.list_blobs(bucket_name)
    for blob in client.list_blobs(bucket_name, prefix=title_prefix):
        if not blob.name.endswith("titleID.pickle"):
            continue
        with blob.open("rb") as f:
            title_id_dict = pickle.load(f)
    for doc_id in doc_ids:
        ret.append((str(doc_id), title_id_dict[doc_id]))
        
    return ret


def backend_search(query):
    k1 = 1.5
    b = 0.5
    body_timeout = 15
    
    if(len(query.split(' ')) > 1):
        title_score = parallel_calculate_BM25_class(k1, b, query, 'title')
        with ThreadPoolExecutor() as executor:
            future = executor.submit(parallel_calculate_BM25_class, k1, b, query, 'body')
            try:
                body_score = future.result(timeout=body_timeout)
            except TimeoutError:
                body_score = []
    
    else:
        title_score = parallel_calculate_BM25_class(k1, b, query, 'title')
        if not title_score:
            body_score = parallel_calculate_BM25_class(k1, b, query, 'body')
        else:
            body_score = []
        
        
    res_score = combine_scores(title_score, body_score)
    ret = get_title([doc for doc, score in res_score])

    return ret
