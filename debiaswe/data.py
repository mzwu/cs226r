import json
import os

"""
Tools for data operations

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""
PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def load_attributes_el():
    attributes_file = os.path.join(PKG_DIR, '../data', 'attributes_el.json')
    with open(attributes_file, 'r', encoding='utf-8') as f:
        attributes = json.load(f)
    print('Loaded attributes\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return attributes

def load_attributes_fr():
    attributes_file = os.path.join(PKG_DIR, '../data', 'attributes_fr.json')
    with open(attributes_file, 'r', encoding='utf-8') as f:
        attributes = json.load(f)
    print('Loaded attributes\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return attributes

def load_attributes_en():
    attributes_file = os.path.join(PKG_DIR, '../data', 'attributes.json')
    with open(attributes_file, 'r', encoding='utf-8') as f:
        attributes = json.load(f)
    print('Loaded attributes\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return attributes
