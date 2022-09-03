import random
import numpy as np
import sys
import os
import re
import json
import struct
import csv
from tensorflow.core.example import example_pb2
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import csv
import json
import pickle

import re, unicodedata
import nltk
import inflect
from nltk import word_tokenize

def read_json(filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

def swap(l,id1,id2):
    e1 = l[id1]
    l[id1] = l[id2]
    l[id2] = e1
    return l

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def preprocessing(sample):
    words = nltk.word_tokenize(sample)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    return words

class data_maker():
    def __init__(self):
        self.word_tokenizer = RegexpTokenizer(r'\w+')
        self.word_id_dict = read_json('giga_word/new_word_id_dict.json')


    def make_pretrain_data(self):
        print("hello world")
        result = []
        source_fp = open('giga_word/pretrain_article.txt','w')
        target_fp = open('giga_word/pretrain_target.txt','w')
        with open('giga_word/NELA_Train.csv', newline='', encoding="utf8") as csvfile:
            reader = csv.reader(csvfile)
            isstart =0
            for row in reader:
                if(isstart == 0):
                    isstart=1
                    continue
                isstart = isstart + 1
                if(isstart==10000):
                    break
                body = preprocessing(row[2])
                headline = preprocessing(row[1])
                target = ""
                source = ""
                for word in body:
                    if word in self.word_id_dict and word != ',' and word != '<unk>':
                        source = source + " " + word

                for word in headline:
                    if word in self.word_id_dict and word != ',' and word != '<unk>':
                        target = target + " " + word

                source_fp.write(source + '\n')
                target_fp.write(target + '\n')


if __name__=='__main__':
    maker = data_maker()
    maker.make_pretrain_data()