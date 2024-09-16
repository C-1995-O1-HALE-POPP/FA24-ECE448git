# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=10, pos_prior=0.95, silently=False):
    print_values(laplace,pos_prior)
    # word_dict = [neg_dict, pos_dict]
    word_dict, dict_size = train(train_set, train_labels,silently)
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(evaluate(doc,laplace,word_dict,pos_prior,dict_size))
    return yhats

def evaluate(doc,laplace,word_dict,pos_prior,dict_size):
    pos_log,neg_log=math.log(pos_prior),math.log(1-pos_prior)
    for word in doc:
        pos_log+=laplaces_smooth(word,word_dict[1],dict_size[1],laplace)
        neg_log+=laplaces_smooth(word,word_dict[0],dict_size[0],laplace)
    if pos_log>neg_log:
        return 1
    else: 
        return 0

def laplaces_smooth(word,dict,num_tokens,laplace):
    types=len(dict.keys())
    if word in dict:
        p=(dict[word]+laplace)/(num_tokens+(1+types)*laplace)
    else:
        p=(laplace)/(num_tokens+(1+types)*laplace)
    return math.log(p)

def train(train_set, train_labels, silently=False):
    word_dict = [{}, {}]
    type_count = [0, 0]
    for label, doc in tqdm(zip(train_labels, train_set), disable=silently):
        for word in doc:
            if word in word_dict[label]:
                word_dict[label][word] += 1
            else:
                word_dict[label][word] = 1
            type_count[label] += 1
    return word_dict, type_count