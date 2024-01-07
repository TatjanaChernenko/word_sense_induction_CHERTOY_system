#!/bin/env python3
# -*- coding: utf-8 -*-

#chertoy.py
#
#usage:
#$ ./chertoy.py < path to the project folder > < /name of the folder with your input data > < /name of your topics.txt file > < /name of your results.txt file > < /name of the output file you want to create in the folder with the project >
# 
#author: <Tatjana Chernenko, Utaemon Toyota>

"""
<CHERTOY> 

------------------- DESCRIPTION -------------------

The pipeline system performs the 17th variant of the system for WSI (word sense induction) task (the Task 11 at SemEval 2013), 
which showed the best performance on the trial data.

The system creates semantic related clusters from the given snippets (the text fragments we get back from the search engine) 
for each pre-defined ambigue topic.

------------------- METHODS -------------------

For the WSI purposes it uses the following methods:

- For pre-rpocessing: tokenization + remove punctuation
- Language model: sense2vec (paper: https://arxiv.org/abs/1511.06388, code: https://github.com/explosion/sense2vec)
- Compositional semantics: vector mixture model (BOW (bag-of-words) representation with summarization for each snippet)
- Clustering: Mean Shift clustering with sklearn.cluster 
(http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift) with default parameters. 

"""
import sys
import sense2vec
from collections import defaultdict, deque
import re 
import math
import numpy as np
from sklearn.cluster import MeanShift

# Read Data; get the number of topics:
def read_data(topics_file, results_file):
    with open(topics_file, "r") as f:
        topics_data = f.readlines()
    with open(results_file, "r") as f:
        results_data = f.readlines()
    number_topics = 0
    for line in topics_data:
        if not line.startswith("ID"):
            number_topics += 1    
    return topics_data, results_data, number_topics

# Create a vocabulary with topics as keys and lists with lists of snippets with IDs as values:
def devide_text_into_topics(results_data, number_topics): 
    text = defaultdict(list) 
    for line in results_data:
        if not line.startswith("ID"):  
            if line.split()[0].split(".")[0] not in text.keys():
                text[line.split()[0].split(".")[0]] = []
                help_structure = []
                help_structure.append(line.split("	")[0])
                help_structure.append(line.split("	")[2:])
                text[line.split()[0].split(".")[0]].append(help_structure) 
            else: 
                help_structure = []
                help_structure.append(line.split("	")[0])
                help_structure.append(line.split("	")[2:])
                text[line.split()[0].split(".")[0]].append(help_structure)
    # Clean sentences from "\n":
    for values in text.values():
        for paragr in values:
            for sent in paragr:
                if sent == "\n":
                    paragr.remove(sent) 
    #print(text["45"]) # example of the output for the topic "45"
    return text
   
# Preprocess Data (tokenize every sentence in every topic):
def preprocess_data(text):
    # Tokenize:
    for value in text.values():
        for paragr in value:
            for i in range(1,len(paragr)): 
                tokens = re.findall(r"\w+", str(paragr[i])) # remove punctuation
                words = [] 
                for word in tokens:
                    if word == "n": 
                        words.append(" ")
                    else: 
                        words.append(word.strip()) # delete first empty placeholders
                paragr[i] = words      
    prepr_data = text
    #print(prepr_data["45"]) # example of the output for the topic "45"
    return prepr_data

# For every word in a sentence make a vector representation with sense2vec; make a compositional vector for every sentence as sum of BOW vectors:
def compos_sense2vec(prepr_data,len_vector):
    model = sense2vec.load()
    for value in prepr_data.values(): 
        for paragr in value: 
            par_list = [] 
            vector_paragr = np.zeros(len_vector) 
            for sent in paragr[1:]: 
                vector_sent = []             
                for word in sent: 
                    try:
                        freq, query_vector = model[word] 
                        vector_sent.append(query_vector) 
                    except:
                        continue
                summe = np.zeros(len_vector) 
                for vector in vector_sent: 
                    summe+=vector 
                par_list.append(summe)
            for sentence in par_list: 
                 vector_paragr+=sentence 
            paragr.append(vector_paragr)            
    compos_vectors = prepr_data
    #print(compos_vectors["45"]) # example of the output for the topic "45"    
    return compos_vectors

# Cluster sentences in every topic with Mean Shift (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift) and create an output file:
def cluster(compos_vectors, output, number_clusters):
    f = open(output, "a")
    f.write("subTopicID"+"	"+"resultID\n") 
    lines = []
    for value in compos_vectors.values():
        z = []
        for sent in value:
            sent_id = sent[0]
            vector = sent[-1]
            z.append(vector)
        all_for_topic = np.array(z)
        meanshift = MeanShift().fit(all_for_topic)
        for i in range(len(meanshift.labels_)):
            one_line = str(value[0][0].split(".")[0])+"."+str(meanshift.labels_[i]) + "	" + (str(value[i][0])+"\n")
            lines.append(one_line)
    sort = sorted([el.split("	") for el in lines],key=lambda i: float(i[0]))
    for i in range(len(sort)):
        sort[i]=str(sort[i][0]+"	"+sort[i][1])
    for el in sort:
        f.write(el)    
    f.close()
    return meanshift

if __name__ == "__main__":
    arg_list = sys.argv
    if len(arg_list) == 6:
        path = arg_list[1]
        trial_path = arg_list[2]
        trial_topics = arg_list[3]
        trial_results = arg_list[4]
        output_name = arg_list[5]
    else:
        path = "/home/tatiana/Desktop/MyPython/Projects/4_Semester/F_Semantik/FSemantik_Projekt/Pool"
        trial_path = "/test"
        trial_topics = "/topics.txt"
        trial_results = "/results.txt"
        output_name = "/output_chertoy.txt"

    topics_data, results_data, number_topics = read_data(str(path)+str(trial_path)+str(trial_topics),str(path)+str(trial_path)+str(trial_results))
    text = devide_text_into_topics(results_data, number_topics)
    prepr_data = preprocess_data(text)
    compos_vectors = compos_sense2vec(prepr_data, 128)
    clusters = cluster(compos_vectors, str(path)+str(output_name), 7)
    print("Done.")

