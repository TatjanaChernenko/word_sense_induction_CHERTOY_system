#!/bin/env python3.5
#variant_25.py
#
#usage:
#$ ./variant_25.py
# 
#author: <Tatjana Chernenko, Utaemon Toyota>

"""
<Variant 25> 

------------------- DESCRIPTION -------------------

The pipeline system performs the 25th variant of the system for WSI (word sense induction) task (the Task 11 at SemEval 2013).

The system creates semantic related clusters from the given snippets (the text fragments we get back from the search engine) for each pre-defined ambigue topic.

------------------- METHODS -------------------

For the WSI purposes it uses the following methods:
- For pre-rpocessing: tokenization step
- Language model: sent2vec, plaintext_unigramm - trained Model (Wikipedia 2017)
- Compositional semantics: vector mixture model (BOW (bag-of-words) representation with summarization for each snippet)
- Clustering: KMeans clustering with sklearn.cluster (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) with the following parameters: 
	n_clusters=given for each topic (read from subTopic.txt), 
	random_state=0. 
Other values are default.

------------------- EVALUATION -------------------

=========== Final average value of F1: =====================
average F1 = 0.6822850886106397

=========== Final average value of Rand Index: =============
average Rand Index = 0.42222222222222217

=========== Final average value of Adjusted Rand Index: ====
average Adj Rand Index = 0.0774543476929948

=========== Final average value of Jaccard Index: ==========
average Jaccard Index = 0.23433892522626892

================ Statistics: ====================================
============ average number of created clusters: 7.5
============ average cluster size: 17.361111111111114


"""
import sys
sys.path.append("/proj/toyota")
import sent2vec
from collections import defaultdict, deque
import re 
import math
import numpy as np
from sklearn.cluster import KMeans

# Read Data; get the number of topics and number of subtopics for each topic:
def read_data(topics_file, subtopics_file, results_file):
    with open(topics_file, "r") as f:
        topics_data = f.readlines()
    with open(subtopics_file, "r") as f:
        subtopics_data = f.readlines()
    with open(results_file, "r") as f:
        results_data = f.readlines()
    number_topics = 0
    for line in topics_data:
        if not line.startswith("ID"):
            number_topics += 1
    subtopics = {}
    for line in subtopics_data:
        if not line.startswith("ID"):
            topic = line.split("	")[0].split(".")[0]
            if topic not in subtopics.keys(): 
                subtopics[topic]=1
            else:
                subtopics[topic]+=1
    return topics_data, subtopics_data, results_data, number_topics, subtopics

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
    # clean sentences from "\n":
    for values in text.values():
        for paragr in values:
            for sent in paragr:
                if sent == "\n":
                    paragr.remove(sent) 
    #print(text["45"]) # example of the output for the topic "45"
    return text 
   
# Preprocess Data (tokenize every sentence in every topic):
def preprocess_data(text):
    for value in text.values():
        for paragr in value:
            for i in range(1,len(paragr)): 
                tokens = re.findall(r"\w+(?:[-']\w+)*|'|[-.(]+|\s\w*", str(paragr[i])) # don't remove punctuation
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
def compos_sent2vec(prepr_data,len_vector):
    model = sent2vec.Sent2vecModel()
    model.load_model("/proj/toyota/wiki_model_unigram.bin")
    for value in prepr_data.values(): 
        for paragr in value: #one snippet
            par_list = [] # list with a snippet
            vector_paragr = np.zeros(len_vector) #vector for one snippet for a sum
            for sent in paragr[1:]: #sent in a snippet
                vector_sent = []     #list for all sentences in a snippet          
                for word in sent: #word
                    try:
                        query_vector = model.embed_sentence(word)
                        vector_sent.append(query_vector) #add a word-vector to a list for sentences in a snippet - BOW for all words in a snippet - now for a sentence - for every sentence
                    except:
                        continue
                summe = np.zeros(len_vector) # vector for a summ 
                for vector in vector_sent: # for one word in all sentences
                    summe+=vector # summ all words in a snippet - vector for a snippet
                par_list.append(summe)#?#add a summ(vector for a snippet) to a list with a snippet
            for sentence in par_list: # for all snippet-vectors
                 vector_paragr+=sentence # sum all snippets
            paragr.append(vector_paragr)             #add to a snippet a summ of all snippets
    compos_vectors = prepr_data
    #print(compos_vectors["45"]) # example of the output for the topic "45"    
    return compos_vectors

# Cluster sentences in every topic with KMeans clustering: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html ) and create an output file:
def cluster(compos_vectors, output, subtopics):
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
        number_clusters = subtopics[sent_id.split(".")[0]]
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
    path = "/proj/toyota/Pool"
    trial_path = "/semeval-2013_task11_trial"
    trial_topics = "/topics.txt"
    trial_subtopics = "/subTopics.txt"
    trial_results = "/results.txt"

    topics_data, subtopics_data, results_data, number_topics, subtopics = read_data(path+trial_path+trial_topics,path+trial_path+trial_subtopics,path+trial_path+trial_results)
    text = devide_text_into_topics(results_data, number_topics)
    prepr_data = preprocess_data(text)
    compos_vectors = compos_sent2vec(prepr_data, 700)
    clusters = cluster(compos_vectors, path+"/output_25.txt", subtopics)
    print("Done.")

