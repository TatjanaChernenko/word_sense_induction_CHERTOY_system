#!/bin/env python3.5
#variant_21.py
#
#usage:
#$ ./variant_21.py
# 
#author: <Tatjana Chernenko, Utaemon Toyota>

"""
<VARIANT 21> 

------------------- DESCRIPTION -------------------

The pipeline system performs the 21th variant of the system for WSI (word sense induction) task (the Task 11 at SemEval 2013).

The system creates semantic related clusters from the given snippets (the text fragments we get back from the search engine) for each pre-defined ambigue topic.

------------------- METHODS -------------------

For the WSI purposes it uses the following methods:
- For pre-rpocessing: tokenization + remove punctuation
- Language model: sense2vec (paper: https://arxiv.org/abs/1511.06388, code: https://github.com/explosion/sense2vec)
- Compositional semantics: vector mixture model (BOW (bag-of-words) representation with summarization for each snippet)
- Clustering: Cos similarity with max Similarity

------------------- EVALUATION -------------------

=========== Final average value of F1: =====================
average F1 = 0.6609591661520388

=========== Final average value of Rand Index: =============
average Rand Index = 0.5332828282828284

=========== Final average value of Adjusted Rand Index: ====
average Adj Rand Index = -0.005157730197548291

=========== Final average value of Jaccard Index: ==========
average Jaccard Index = 0.48910264449366275

================ Statistics: ====================================
============ average number of created clusters: 3.25
============ average cluster size: 33.333333333333336


"""
import sense2vec
from collections import defaultdict, deque
import re 
import math
import numpy as np
from sklearn.cluster import MeanShift
import nltk
from scipy import spatial

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

    # POS-Tagging with nltk universal Tagset:
    for value in text.values():
        for paragr in value:
            for i in range(1,len(paragr)): 
                tagged = nltk.pos_tag(paragr[i],tagset='universal')
                for k in range(len(tagged)):
                    if tagged[k][1] == "PRT":
                        tagged[k][1] == "PART"
                    if tagged[k][1] == ".":
                        tagged[k][1] == "PUNCT"
                    tagged[k] = tagged[k][0]+"|"+tagged[k][1]               
                paragr[i] = tagged 
    prepr_data = text
    #print(prepr_data["45"]) # example of the output for the topic "45"
    return prepr_data

# For every word in a sentence make a vector representation with sense2vec; make a compositional vector for every sentence as sum of BOW vectors:
def compos_sense2vec(prepr_data,len_vector):
    model = sense2vec.load()
    for value in prepr_data.values(): 
        for paragr in value: #one snippet
            par_list = [] # list with a snippet
            vector_paragr = np.zeros(len_vector) #vector for one snippet for a sum
            for sent in paragr[1:]: #sent in a snippet
                vector_sent = []     #list for all sentences in a snippet          
                for word in sent: #word
                    try:
                        freq, query_vector = model[word] #get a vector for the word
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

# Create a vocabulary for subtopics with topics as keys and lists with subtopics with IDs as values:
def devide_subtopics_into_topics(subtopics_data, number_topics): 
    subtopics_vectors = defaultdict(list) 
    for line in subtopics_data:
        if not line.startswith("ID"):  
            if line.split()[0].split(".")[0] not in subtopics_vectors.keys():
                subtopics_vectors[line.split()[0].split(".")[0]] = []
                help_structure = []
                help_structure.append(line.split("	")[0])
                help_structure.append(line.split("	")[1])
                subtopics_vectors[line.split()[0].split(".")[0]].append(help_structure) 
            else: 
                help_structure = []
                help_structure.append(line.split("	")[0])
                help_structure.append(line.split("	")[1])
                subtopics_vectors[line.split()[0].split(".")[0]].append(help_structure)
    # Clean sentences from "\n":
    for values in subtopics_vectors.values():
        for paragr in values:
            for sent in paragr:
                if sent == "\n":
                    paragr.remove(sent) 
    #print("Subtopics Vectors: ", subtopics_vectors) 
    return subtopics_vectors

# Make vector representation of Subtopics with sense2vec, sum of BOW with sense2vec:
def compose_sense2vec_subtopics(subtopics_vectors, len_vector):
    model = sense2vec.load()
    for value in subtopics_vectors.values(): 
        for paragr in value: #one snippet
            par_list = [] # list with a snippet
            vector_paragr = np.zeros(len_vector) #vector for one snippet for a sum
            for sent in paragr[1:]: #sent in a snippet
                vector_sent = []     #list for all sentences in a snippet          
                for word in sent: #word
                    try:
                        freq, query_vector = model[word] #get a vector for the word
                        vector_sent.append(query_vector) #add a word-vector to a list for sentences in a snippet - BOW for all words in a snippet - now for a sentence - for every sentence
                    except:
                        continue
                summe = np.zeros(len_vector) # vector for a summ 
                for vector in vector_sent: # for one word in all sentences
                    summe+=vector # summ all words in a snippet - vector for a snippet
                    #par_list.append(summe) #?!!!!!! war
                par_list.append(summe)#?#add a summ(vector for a snippet) to a list with a snippet
            for sentence in par_list: # for all snippet-vectors
                 vector_paragr+=sentence # sum all snippets
            paragr.append(vector_paragr)             #add to a snippet a summ of all snippets
    compos_subtopics_vectors = subtopics_vectors
    #print("Composed Subtopics Vectors: ", compos_subtopics_vectors) 
    return compos_subtopics_vectors


# Create a vocabulary with cos similarities to subtopics for every snippet:
def cos_sim_vocab(compos_subtopics_vectors, compos_vectors):
    all_sim = {}
    for value in compos_vectors.values(): #value = all sent with metainfo for one topic
        #print("\n\nTOPIC: ", value[0][0].split(".")[0], "\n")
        for snippet in value: #snippet for one topic (   ['47.100', ['The|DET', ...], array([-2.11360547e+00, ...]] )
            similarities = {}
            #print("\nsnippet id: ", snippet[0])
            for all_subt in compos_subtopics_vectors[value[0][0].split(".")[0]]:
                 sim = 1 - spatial.distance.cosine(snippet[-1], all_subt[2])
                 a = []
                 a.append(all_subt[0])
                 a.append(sim)	        
                 if snippet[0] not in similarities.keys():
                     similarities[snippet[0]] = []
                     similarities[snippet[0]].append(a)
                 else:
                     similarities[snippet[0]].append(a)
            if value[0][0].split(".")[0] not in all_sim.keys():
                all_sim[value[0][0].split(".")[0]] = []
                all_sim[value[0][0].split(".")[0]].append(similarities)
            else:
                all_sim[value[0][0].split(".")[0]].append(similarities)
    #print("\nAll similarities: ", all_sim)
    return all_sim

#LIKE WSD: Cluster sentences (snippets) based on cos similarity to Subtopics without sim_factor (use max cos sim); create an output file:
def cos_sim_clustering(cos_sim_vocab, output):
    f = open(output, "a")
    f.write("subTopicID"+"	"+"resultID\n") 
    lines = []
    result = {}
    for topic in cos_sim_vocab.keys(): # 46
        #print("TOPIC: ", topic)
        if topic not in result.keys():
            result[topic] = []
        for snippet in cos_sim_vocab[topic]:# (list with) one vocab with keys=snippet ids
            voc = {}
            max_sim = np.float(0)
            max_id = "" 
            for simil in snippet.values():
                for el in simil:
                     if el[1] > max_sim:
                        max_sim = el[1] 
                        max_id = el[0]
            max_value = []
            structure = []
            structure.append(max_id)
            structure.append(max_sim)
            max_value.append(structure)
            for el in snippet.keys():
                if el not in voc.keys():
                    voc[el] = []
                    voc[el].append(max_value)
                else:
                    voc[el].append(max_value)
            result[topic].append(voc)
    for value in result.values():
        for el in value:
            for e in el.keys():
                one_line = str(el[e][0][0][0]) + "	" + (str(e)+"\n")
                lines.append(one_line)

    sort = sorted(lines)
    #print(sort)
    for el in sort:
        f.write(el)    
    f.close()
    return result

# Cluster sentences in every topic with Mean Shift (http://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift) and create an output file:
def cluster(compos_vectors, output, number_clusters):
    f = open(output, "a")
    f.write("subTopicID"+"	"+"resultID\n") 
    lines = []
    for value in compos_vectors.values():
        #print("\n\nTOPIC: ", value[0][0].split(".")[0], "\n")
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
    sort = sorted(lines)
    for el in sort:
        f.write(el)    
    f.close()
    return meanshift

if __name__ == "__main__":
    path = "/home/tatiana/Desktop/MyPython/Projects/4_Semester/F_Semantik/FSemantik_Projekt/Projekt1"
    trial_path = "/semeval-2013_task11_trial"
    trial_topics = "/topics.txt"
    trial_subtopics = "/subTopics.txt"
    trial_results = "/results.txt"

    topics_data, subtopics_data, results_data, number_topics, subtopics = read_data(path+trial_path+trial_topics,path+trial_path+trial_subtopics,path+trial_path+trial_results)
    text = devide_text_into_topics(results_data, number_topics)
    prepr_data = preprocess_data(text)
    compos_vectors = compos_sense2vec(prepr_data, 128)
    subtopics_vectors = devide_subtopics_into_topics(subtopics_data, number_topics)
    compos_subtopics_vectors = compose_sense2vec_subtopics(subtopics_vectors, 128)
    cos_sim_vocab = cos_sim_vocab(compos_subtopics_vectors, compos_vectors)
    clusters = cos_sim_clustering(cos_sim_vocab, path+"/output_21.txt")
    print("Done.")


