# AUTHORS

Chernenko Tatjana, Toyota Utaemon

2017/2018

Institute of Computational Linguistics

Heidelberg University, Germany

# CHERTOY

Keywords: WSI · Word Sense Induction · Data processing · Web search result clustering · CHERTOY

CHERTOY is an implementation of the CHERTOY system designed by Tatjana Chernenko and Utaemon Toyota for the Word Sense Induction task (Task 11 at SemEval 2013). 
This project encompasses the baseline system and 40 experimental variations.

In the resulting [paper](https://github.com/TatjanaChernenko/word_sense_induction_CHERTOY_system/blob/main/Results.pdf) we provide an approach to improve word sense induction systems (WSI) 
for web search result clustering.

Our research delves into exploring language models, specific feature sets, and 
clustering algorithms based on sense2vec and sent2vec. Through 
meticulous experimentation across 40 variants, we have gained insightful observations 
on the effects of various feature combinations. 
These insights have culminated in the development of our Word Sense Induction (WSI) 
system, CHERTOY.

The system's primary objective is to generate semantically related 
clusters from provided text snippets, sourced from search engine results, 
for each predefined ambiguous topic. It undertakes data preprocessing, constructs 
language models using vector representations for individual snippets using sense2vec, 
employs a vector mixture model (BOW representation with summarization for each snippet), and generates 
semantic clusters through the Mean Shift clustering algorithm.

## RUNNING INSTRUCTIONS

Dependencies:
- sense2vec (paper: https://arxiv.org/abs/1511.06388, code: https://github.com/explosion/sense2vec)
- skikit-learn

### Input files:

The input data must consist of two text files: results.txt and topics.txt.

* results.txt

A file with snippets in the following format: 

ID \t url \t title \t snippet

There are no empty lines between the snippets.

_Example:_

ID \t url \t title \t snippet

1.1	\t http://www.polaroid.com/	\t Polaroid | Home | 74.208.163.206	Create and share like never before at <b>Polaroid</b>.com. Find instant film and   cameras reinvented for the digital age. Plus, digital cameras, digital camcorders,   LCD <b>...</b>

1.2	\t http://www.polaroid.com/products	\t products | www.polaroid.com	Come check out a listing of Polaroid products, by category.

1.3	\t http://en.wikipedia.org/wiki/Polaroid_ \t Corporation	Polaroid Corporation - Wikipedia, the free encyclopedia	<b>Polaroid</b> Corporation is an American-based international consumer electronics   and eyewear company, originally founded in 1937 by Edwin H. Land. It is most <b>...</b>


* topics.txt

A file with ambiguous topics. The system will create clusters for each of these topics.

_Example:_

id	description

1	polaroid

2	kangaroo

3	shakira

4	kawasaki


### Output files:

CHERTOY produces the output file (output.txt) that is formatted as follows:

subTopicID \t resultID

Here the subTopicID consists of the topic ID from the file topicx.txt and the number of the cluster (meaning). 
The resultID is the ID number of the snippet from the file results.txt.

_Example:_

subTopicID \t resultID

45.0 \t 45.1

45.0 \t 45.10

45.1 \t 45.20

### Create a folder structure:

Create a folder with your projects:
* project folder
    * folder with your input files (resutls.txt and topics.txt)
 
After running the system you'll have the output file in your project folder.

### RUN THE SYSTEM:


git clone https://gitlab.cl.uni-heidelberg.de/semantik_project/wsi_chernenko_schindler_toyota.git

cd bin

python3 chertoy.py < path to the project folder > < /name of the folder with your input data > < /name of your topics.txt file > < /name of your results.txt file > < /name of the output file you want to create in the folder with the project > 

_Example:_

python3 chertoy.py /home/tatiana/Desktop/FSemantik_Projekt /test /topics.txt /results.txt /output_chertoy.txt


### Other files:

* Performances_Table.pdf - a performance table with F1, RI, ARI and JI values of the baseline and 40 experiments (incl. CHERTOY) on the trial data.

* bin

The folder contains the implementation of the CHERTOY system.

* experiments

The folder experiments contains an implementation of the baseline and 40 different experiments to improve its performance.

* lib

The folder contains code for preprocessing Wikipedia Dataset to train own sent2vec models for the experiments and a README file. Our preprocessed Wikipedia 2017 dataset and two self-trained models of the Wikipedia 2017 dataset, that we used in our experiments with sent2vec, are provided on /proj/toyota on the server of the Institut of Computerlinguistics Heidelberg.
Other models that we used during our experiments can be found in sense2vec and sent2vec repositories.

* experiments

Implementation of the baseline and 40 experiments with it.

* output

outputs\_trial_data - output files for the experiments on the trial data
output\_test\_data - output file for the test data

### LICENSES

This software is distributed under the MIT License.

The part of the system utilizes sense2vec, which is distributed under the following License:


The MIT License (MIT)

Copyright (C) 2016 spaCy GmbH
              2016 ExplosionAI UG (haftungsbeschränkt)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

