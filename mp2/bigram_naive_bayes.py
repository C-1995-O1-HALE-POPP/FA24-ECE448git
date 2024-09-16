# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.005, bigram_laplace=0.005, bigram_lambda=0.5, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    single_dict, single_num = single_train(train_set, train_labels, silently)
    pair_dict, pair_num = pair_train(train_set, train_labels, silently)

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(evaluate(doc, pos_prior, bigram_lambda,
                              single_dict, single_num, unigram_laplace,
                              pair_dict, pair_num, bigram_laplace))

    return yhats

def evaluate(doc, pos_prior, bigram_lambda,
            single_dict, single_num, unigram_laplace,
            pair_dict, pair_num,  bigram_laplace):
    
    single_pos_log,single_neg_log=math.log(pos_prior),math.log(1-pos_prior)
    pair_pos_log,pair_neg_log=single_pos_log,single_neg_log
    
    for word in doc:
        single_pos_log+=laplaces_smooth(word,single_dict[1],single_num[1],unigram_laplace)
        single_neg_log+=laplaces_smooth(word,single_dict[0],single_num[0],unigram_laplace)
    for i in range(len(doc)-1):
        pair=(doc[i],doc[i+1])
        pair_pos_log+=laplaces_smooth(pair,pair_dict[1],pair_num[1],bigram_laplace)
        pair_neg_log+=laplaces_smooth(pair,pair_dict[0],pair_num[0],bigram_laplace)
    pos_log=bigram_lambda*pair_pos_log+(1-bigram_lambda)*single_pos_log
    neg_log=bigram_lambda*pair_neg_log+(1-bigram_lambda)*single_neg_log

    if pos_log>neg_log:
        return 1
    else: 
        return 0

def laplaces_smooth(word,dict,num_tokens,laplace):
    types=len(dict)
    if word in dict:
        p=(dict[word]+laplace)/(num_tokens+(1+types)*laplace)
    else:
        p=(laplace)/(num_tokens+(1+types)*laplace)
    return math.log(p)

def single_train(train_set, train_labels, silently=False):
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

def pair_train(train_set, train_labels, silently=False):
    pair_dict = [{}, {}]
    type_count = [0, 0]
    for label, doc in tqdm(zip(train_labels, train_set), disable=silently):
        for i in range(len(doc)-1):
            pair = (doc[i], doc[i+1])
            if pair in pair_dict[label]:
                pair_dict[label][pair] += 1
            else:
                pair_dict[label][pair] = 1
            type_count[label] += 1
    return pair_dict, type_count