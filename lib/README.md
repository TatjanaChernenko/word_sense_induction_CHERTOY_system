# CHERTOY - Creating language model with sent2vec

This is an implementation to provide necessary preprocessing steps for modeling an own sent2vec model which is used in the experiments. The two language models we built are a uni-gram and a bi-gram model over the wikipedia 2017 corpus.

## RUNNING INSTRUCTIONS

## Preprocessing Wikipedia Dump

Download Wikipedia Dump
- Wikipedia Dumps for the english language is provided on https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia
- For our model we used enwiki-20170820-pages-articles-multistream.xml.bz2 (14.1 GiB)

Dependencies:
- wikiExtractor: http://attardi.github.io/wikiextractor
- fasttext: https://github.com/facebookresearch/fastText
- sent2vec: https://github.com/epfml/sent2vec

First the Wikipedia text needs to be extracted from the provided XML.
-extracted file: enwiki-20170820-pages-articles-multistream.xml (21.0GB)

From the XML the plain text will be extracted using wikiExtractor:
WikiExtractor.py -o OUTPUT-DIRECTORY INPUT-XML-FILE

_Example_
WikiExtractor.py -o /wikitext enwiki-20170820-pages-articles-multistream.xml

WikiExtractor will create several directories AA, AB, AC, ...,  CH with a total size of 6.2GB. Each directory contains 100 .txt documents (besides CH -> 82).
Each article begins with an ID such as <doc id="12" url="https://en.wikipedia.org/wiki?curid=12" title="Anarchism">. Also comments in Parentheses are provided.
Using preprocess_wikitext.py we delete all IDs, parentheses with their content and also quotes like ' or " for getting a plain wikipedia text. The output text file contains one sentence per line. 

_Usage_
python3 preprocess_wikitext.py input_directory_path output_txt_file_path

_Examle_
python3 preprocess_wikitext.py /home/utaemon/Semantik_Projekt/results/ /home/utaemon/Semantik_Projekt/plain/

The text file are organized in the same way as in the input text files in directories AA, AB...
To collect all texts into one file collect_all_wiki_in_one.py can be used. In our case the output file will have a total size of 4.1GB.

_Usage_
python3 preprocess_wikitext.py input_directory_path output_dir_file_path

_Example_
python3 preprocess_wikitext.py /home/utaemon/Semantik_Projekt/plain/ /home/utaemon/Semantik_Projekt/all_plain_texts.txt


## Create new sent2vec model

Move to the sent2vec directory. Here you can run following instruction in the terminal:

For Uni-grams:
./fasttext sent2vec -input /proj/toyota/all_plain_texts.txt -output /proj/toyota/wiki_model_unigram -minCount 1 -dim 700 -epoch 10 -lr 0.2 -wordNgrams 1 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000

For Bi-grams:
./fasttext sent2vec -input /proj/toyota/all_plain_texts.txt -output /proj/toyota/wiki_model_bigram -minCount 1 -dim 700 -epoch 10 -lr 0.2 -wordNgrams 2 -loss ns -neg 10 -thread 20 -t 0.000005 -dropoutK 4 -minCountLabel 20 -bucket 4000000

In our case it will make a model over 321 million words and containing 4518148 number of words.

### Output models:
Both models are provided on /proj/toyota on the server of the Institute of Computer Linguistics Heidelberg.

Uni-gram model:
wiki_model_unigram.bin (25.4GB)

Bi-gram model:
wiki_model_bigram.bin (36.6GB)


### LICENSES 
# wikiExtractor
GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

 Copyright (C) 2007 Free Software Foundation, Inc. <http://fsf.org/>
 Everyone is permitted to copy and distribute verbatim copies
 of this license document, but changing it is not allowed.

# fasttext
BSD License

For fastText software

Copyright (c) 2016-present, Facebook, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

 * Neither the name Facebook nor the names of its contributors may be used to
   endorse or promote products derived from this software without specific
   prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# sent2vec (for code and pre-trained models)
Matteo Pagliardini, Prakhar Gupta, Martin Jaggi, Unsupervised Learning of Sentence Embeddings using Compositional n-Gram Features NAACL 2018.

