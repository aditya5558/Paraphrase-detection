import numpy as np
import tensorflow as tf
import nltk
import pandas as pd
import os
import _pickle as cPickle

class model:


	def __init__(self):

		self.word2idx = {'PAD' : 0}
		self.weights = []
		
		self.features = {}
		self.word_list = []

		self.glove_path = '/home/aditya/Desktop/paraphrasing/glove.840B.300d.txt'
		self.dataset_path = '/home/aditya/Desktop/paraphrasing/msr-paraphrase-corpus/msr_paraphrase_train.txt'

		


	def load_glove(self):

		glove_cache_file = os.path.join('cache', 'glove.pkl')
		word2idx_cache_file = os.path.join('cache', 'word2idx.pkl')

		if os.path.isfile(glove_cache_file):
			print('Loading glove embeddings from : ' + glove_cache_file)
			with open(glove_cache_file, 'rb') as f:
				self.weights = cPickle.load(f)
			print('Done')

			print('Loading word2idx from : ' + word2idx_cache_file)
			with open(word2idx_cache_file, 'rb') as f:
				self.word2idx = cPickle.load(f)
			print('Done')

			self.embed_dim = len(self.weights[0])
			self.vocab_size = self.weights.shape[0]
			# self.word2idx['UNK'] = len(self.weights) - 1

		else:
            
			print('Creating glove embeddings:')

			with open(self.glove_path,'r') as file:
				
				for index, line in enumerate(file):
					values = line.split()
					word = values[0]
				
					try:
						word_weights = np.asarray(values[1:],dtype=np.float32)
						if(word_weights.shape[0] == 300):
							self.word2idx[word] = index+1
							self.weights.append(word_weights)

					except ValueError:
						print('Error at line ',index)

					if index == 100000:
						break

			self.embed_dim = len(self.weights[0])
			self.weights.insert(0,np.random.randn(self.embed_dim))

			self.word2idx['UNK'] = len(self.weights)
			self.weights.append(np.random.randn(self.embed_dim))

			# X = self.weights[0]

			# for i in range(1,len(self.weights)):
			# 	X = np.concatenate((X,self.weights[i]))
			# self.weights = np.concatenate(self.weights,axis=0)
			
			self.weights = np.stack(self.weights)
			
			# self.weights = np.asarray(self.weights)
			# k = self.weights[0].shape[0]
			# for i in range(len(self.weights)):

			# 	if(self.weights[i].shape[0]!=k):
			# 		print (self.weights[i].shape[0],i)
				
			# 		# break
			# # print(self.weights[52344])
			# exit(0)
			
			self.vocab_size = self.weights.shape[0]

			print('Saving word2idx to: ' + word2idx_cache_file)
			with open(word2idx_cache_file, 'wb') as f:
				cPickle.dump(self.word2idx,f)
			print('Done!')

			print('Saving glove embeddings to: ' + glove_cache_file)
			with open(glove_cache_file, 'wb') as f:
				cPickle.dump(self.weights,f)
			print('Done!')
           

		print(self.vocab_size,self.embed_dim)

		
		print("Shape of glove embeddings:",self.weights.shape)
		# print(self.weights)



	def load_dataset(self):

		with open(self.dataset_path,'r') as file1:
			
			for index, line in enumerate(file1):
				
				self.sentence_one = []
				self.sentence_two = []

				values = line.split("\t")
				
				#Sentence 1
				words = values[3]
				self.sentence_one.extend([self.word2idx.get(word,self.word2idx['UNK']) for word in words])

				#Sentecne 2
				words = values[4]
				self.sentence_two.extend([self.word2idx.get(word,self.word2idx['UNK']) for word in words])
				
				break


		# self.features['word_indices'] = self.word_list

		print("Sentence 1 indices:",self.sentence_one)
		print("Sentence 2 indices:",self.sentence_two)

		self.sentence_one_len = len(self.sentence_one)
		self.sentence_two_len = len(self.sentence_two)

		print("Sentence 1 Length:",self.sentence_one_len)
		print("Sentence 2 Length:",self.sentence_two_len)


	def context_layer(self):

		
		embedding_weights = tf.get_variable(
			name = 'embedding_weights',
			shape = (self.vocab_size,self.embed_dim),
			initializer = tf.constant_initializer(self.weights),
			trainable = False)


		embedded_sentence_one = tf.nn.embedding_lookup(embedding_weights,self.sentence_one)
		embedded_sentence_two = tf.nn.embedding_lookup(embedding_weights,self.sentence_two)



		context_rnn_hidden_size = 100

		sentence_enc_fw = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)
		sentence_enc_bw = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)

		# Sentence 1

		outputs_1, states_1  = tf.nn.bidirectional_dynamic_rnn(
		    cell_fw=sentence_enc_fw,
		    cell_bw=sentence_enc_bw,
		    dtype=tf.float32,
		    sequence_length=[self.sentence_one_len],
		    inputs=tf.expand_dims(embedded_sentence_one,0))

		output_fw_1, output_bw_1 = outputs_1
		states_fw_1, states_bw_1 = states_1

		tf.get_variable_scope().reuse_variables()

		#Sentence 2

		outputs_2, states_2  = tf.nn.bidirectional_dynamic_rnn(
		    cell_fw=sentence_enc_fw,
		    cell_bw=sentence_enc_bw,
		    dtype=tf.float32,
		    sequence_length=[self.sentence_two_len],
		    inputs=tf.expand_dims(embedded_sentence_two,0))
		 
		output_fw_2, output_bw_2 = outputs_2
		states_fw_2, states_bw_2 = states_2


		sess = tf.Session()
		sess.run(tf.global_variables_initializer())

		# print(sess.run(output_fw))
		[output_fw_1,output_bw_1,output_fw_2,output_bw_2] = sess.run([output_fw_1,output_bw_1,output_fw_2,output_bw_2])

		print("Sentence 1:")
		print("Forward RNN Output:",output_fw_1.shape,"Backward RNN output:",output_bw_1.shape)

		print("Sentence 2:")
		print("Forward RNN Output:",output_fw_2.shape,"Backward RNN output:",output_bw_2.shape)


if __name__ == '__main__':
	
	x = model()
	x.load_glove()
	x.load_dataset()
	x.context_layer()
