# Paraphrase-detection
Minor Project repository - 6th sem

## Objective
To develop a hybrid model for clinical paraphrase detection.

## Base Model 
A bilateral multi-perspective matching (BiMPM) model is used. Given two sentences P
and Q, the model first encodes them with a BiLSTM encoder. Next, the two encoded
sentences are matched in two directions P against Q and Q against P . In each matching direction, each time
step of one sentence is matched against all time steps of the other sentence from multiple perspectives. Then, another BiLSTM layer is utilized to aggregate the matching results into a fixed-length
matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer.

## Attention Model 

Attention layer extracts words that are important to the meaning of the sentence and aggregate the
representation of those informative words to form a sentence vector. This Model ”attends” to important parts of the sentence.

## Datasets used

### MSRP Dataset

The dataset consists of 5,801 sentence pairs. The average sentence length is 21, the shortest sentence has 7 words and the longest 36. 3,900 are labeled as being in the paraphrase relationship. Standard split of 4,076 training
pairs (67.5 of which are paraphrases) and 1,725 test pairs (66.5 paraphrases) used.

### Medical Dataset

Created a medical paraphrase corpus from the clinical notes in i2b2 dataset.
* Training Set : 150 pairs of sentences
* Test Set : 60 pairs of sentences

### Download links for pretrained word embeddings :

Glove : https://nlp.stanford.edu/projects/glove/

Word2vec : https://code.google.com/archive/p/word2vec/

