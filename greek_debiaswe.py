from __future__ import print_function, division
from matplotlib import pyplot as plt
import json
import random
import numpy as np
from gensim.models import KeyedVectors

import debiaswe as dwe
import debiaswe.we as we
from debiaswe.we import WordEmbedding
from debiaswe.data import load_attributes_el
from debiaswe.debias import debias

# load subtitle word embeddings
E = WordEmbedding('./data/greek_subs.vec')

# load attributes
attributes = load_attributes_el()
attributes_words = [p for p in attributes]

# Define gender direction
v_gender = E.diff("αυτή", "αυτός")

# Load anology pairs
analogies_pair=set()

with open('./data/greek_analogies.tsv', "r", encoding = "utf-8") as f:
    for line in f:
        if "#" in line:
            continue
        l = line.replace("\n", "").split("\t")
        analogies_pair.add((l[0], l[1]))
        analogies_pair.add((l[2], l[3]))

# Function to evaluate prediction accuracies for analogies based on embeddings
def evaluate_w2c(E, analogies_pair):
    tmp_file='./data/temp.bin'
    E.save_w2v(tmp_file)
    # print ("saved")
    tmp_file='./data/temp.bin'
    word_vectors = KeyedVectors.load_word2vec_format(tmp_file, binary=True)

    len_= len(analogies_pair)
    acc = 0
    for p in analogies_pair:
        male=p[0]
        fema=p[1]
        # word vectors
        if (male in word_vectors and fema in word_vectors):
            x = word_vectors.most_similar(positive=['άνδρας', male], negative=['γυναίκα'])[0][0]
            print (male+"="+x+"-"+fema)
            if x.lower() == fema.lower() or male.lower() == x.lower():
                acc+=1

    out = acc/float(len_)
    print (acc)
    print (len_)
    print ("accuracy =",out)
    return out

# Evaluate analogies on original embeddings
evaluate_w2c(E, analogies_pair)

# Load definitional, equalizer, and gender specific words
with open('./data/definitional_el.json', "r", encoding = "utf-8") as f:
    defs = json.load(f)
print("definitional", defs)

with open('./data/equalize_el.json', "r", encoding = "utf-8") as f:
    equalize_pairs = json.load(f)

with open('./data/gender_specific_el.json', "r", encoding = "utf-8") as f:
    gender_specific_words = json.load(f)
print("gender specific", len(gender_specific_words), gender_specific_words[:10])

# Debias embeddings based on the above data
E_new = debias(E, gender_specific_words, defs, equalize_pairs)

# Evaluate analogies based on debiased embeddings
evaluate_w2c(E_new, analogies_pair)