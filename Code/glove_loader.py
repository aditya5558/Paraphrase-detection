import numpy as np
import tensorflow as tf
import nltk
import pandas as pd

##################################################

# Part 1 : Defining word2idx mapping and loading glove corpus

word2idx = {'PAD' : 0}
weights = []

with open('/home/aditya/Desktop/paraphrasing/glove.840B.300d.txt','r') as file:
	
	for index, line in enumerate(file):
		values = line.split()
		word = values[0]
		# print(word)
		try:
			word_weights = np.asarray(values[1:],dtype=np.float32)
			word2idx[word] = index+1
			weights.append(word_weights)
		except ValueError:
			print('Error at line ',index)
		if index == 50000:
		 	break

embed_dim = len(weights[0])
weights.insert(0,np.random.randn(embed_dim))

word2idx['UNK'] = len(weights)
weights.append(np.random.randn(embed_dim))


weights = np.asarray(weights)
vocab_size = weights.shape[0]

# print(word2idx['UNK'])

print(weights.shape)


######################################################

# Part 2 : Loading MSRP corpus and generating word tokens from it to be looked up

features = {}
# features['word_indices'] = nltk.word_tokenize('hello world') 
# features['word_indices'] = [word2idx.get(word,len(weights)) for word in features['word_indices']]


# f = pd.read_csv('/home/aditya/Desktop/paraphrasing/msr-paraphrase-corpus/msr_paraphrase_train.txt')

word_list = []

with open('/home/aditya/Desktop/paraphrasing/msr-paraphrase-corpus/msr_paraphrase_train.txt','r') as file1:
	
	for index, line in enumerate(file1):
		values = line.split()
		words = values[3:]
		# print(words)
		word_list.extend([word2idx.get(word,word2idx['UNK']) for word in words])
		# print(word_list)

features['word_indices'] = word_list

print(len(features['word_indices']))

######################################################

# Part 3 : Looking up all word tokens from MSRP corpus in embedding matrix

glove_constant_initializer = tf.constant_initializer(weights)

embedding_weights = tf.get_variable(
	name = 'embedding_weights',
	shape = (vocab_size,embed_dim),
	initializer = glove_constant_initializer,
	trainable = False)

embedding = tf.nn.embedding_lookup(embedding_weights,features['word_indices'])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

result = sess.run(embedding)
result = np.asarray(result)
print(result.shape)
