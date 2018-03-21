# Paraphrase-detection
Minor Project repository - 6th sem

## Objective
To develop a hybrid model for clinical paraphrase detection in doctors' prescriptions.

## Current Model Overview
A bilateral multi-perspective matching (BiMPM) model is used. Given two sentences P
and Q, the model first encodes them with a BiLSTM encoder. Next, the two encoded
sentences are matched in two directions P against Q and Q against P . In each matching direction, each time
step of one sentence is matched against all time steps of the other sentence from multiple perspectives. Then, another BiLSTM layer is utilized to aggregate the matching results into a fixed-length
matching vector. Finally, based on the matching vector, a decision is made through a fully connected layer.

### Download links for pretrained word embeddings :

Glove : https://nlp.stanford.edu/projects/glove/

Word2vec : https://code.google.com/archive/p/word2vec/

Fasttext : https://github.com/icoxfog417/fastTextJapaneseTutorial
