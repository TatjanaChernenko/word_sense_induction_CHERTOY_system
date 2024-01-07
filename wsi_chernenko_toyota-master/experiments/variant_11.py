#!/bin/env python3.5
#variant_11.py
#
#usage:
#$ ./variant_11.py
# 
#author: <Tatjana Chernenko, Utaemon Toyota>

"""
<VARIANT 11> 

------------------- DESCRIPTION -------------------

The pipeline system performs the 11th variant of the system for WSI (word sense induction) task (the Task 11 at SemEval 2013).

The system creates semantic related clusters from the given snippets (the text fragments we get back from the search engine) for each pre-defined ambigue topic.

------------------- METHODS -------------------

For the WSI purposes it uses the following methods:
- For pre-rpocessing: tokenization + remove punctuation
- Language model: word2vec (with gensim.models.word2vec) trained with Brown Corpus
- Compositional semantics: vector mixture model (BOW (bag-of-words) representation with summarization for each snippet)
- Clustering: KMeans clustering with sklearn.cluster (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) with the following parameters: 
	n_clusters=7, 
	random_state=0. 
Other values are default.

------------------- EVALUATION -------------------

=========== Final average value of F1: =====================
average F1 = 0.6749557232429892

=========== Final average value of Rand Index: =============
average Rand Index = 0.4022727272727273

=========== Final average value of Adjusted Rand Index: ====
average Adj Rand Index = 5.722615092542925E-4

=========== Final average value of Jaccard Index: ==========
average Jaccard Index = 0.20594468165943697

================ Statistics: ====================================
============ average number of created clusters: 7.0
============ average cluster size: 14.285714285714286


"""
from gensim.models.word2vec import Word2Vec
from collections import defaultdict, deque
import re 
import math
import numpy as np
from sklearn.cluster import KMeans
from nltk.corpus import brown

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
                        words.append(word.strip()) # delete first empty placeholder
                paragr[i] = words  
    
    prepr_data = text
    #print(prepr_data["45"]) # example of the output for the topic "45"
    return prepr_data

# For every word in a sentence make a vector representation with sense2vec; make a compositional vector for every sentence as sum of BOW vectors:
def compos_word2vec(prepr_data,len_vector):
    model = Word2Vec(brown.sents(), min_count=1, size=200, iter=40)    
    for value in prepr_data.values():
        for paragr in value:
            par_list = []
            vector_paragr = np.zeros(len_vector)
            for sent in paragr[1:]:
                vector_sent = []               
                for word in sent:
                    try:
                        q_vector = model[word]
                        vector_sent.append(q_vector)
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

# Cluster sentences in every topic with KMeans clustering: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html ) and create an output file:
def cluster(compos_vectors, output, number_clusters):
    f = open(output, "a")
    f.write("subTopicID"+"	"+"resultID\n") 
    lines = []
    for value in compos_vectors.values():
        # print("\n\nTOPIC: ", value[0][0].split(".")[0], "\n")
        z = []
        for sent in value:
            sent_id = sent[0]
            vector = sent[-1]
            z.append(vector)
        all_for_topic = np.array(z)
        kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(all_for_topic)
        for i in range(len(kmeans.labels_)):
            one_line = str(value[0][0].split(".")[0])+"."+str(kmeans.labels_[i]) + "	" + (str(value[i][0])+"\n")
            lines.append(one_line)
    sort = sorted(lines)
    for el in sort:
        f.write(el)    
    f.close()
    return kmeans

if __name__ == "__main__":
    path = "/home/tatiana/Desktop/MyPython/Projects/4_Semester/F_Semantik/FSemantik_Projekt/Projekt1"
    trial_path = "/semeval-2013_task11_trial"
    trial_topics = "/topics.txt"
    trial_results = "/results.txt"

    topics_data, results_data, number_topics = read_data(path+trial_path+trial_topics,path+trial_path+trial_results)
    text = devide_text_into_topics(results_data, number_topics)
    prepr_data = preprocess_data(text)
    compos_vectors = compos_word2vec(prepr_data, 200)
    clusters = cluster(compos_vectors, path+"/output_11.txt", 7)
    print("Done.")

