import numpy as np
import tensorflow as tf
#import nltk
import pandas as pd
import os
import cPickle as cPickle
from matching import bidirectional_match,get_last_output

def one_hot(batch_size,Y):

    B = np.zeros((batch_size,2))

    B[np.arange(batch_size),Y] = 1

    return B.astype(int)

class model:


	def __init__(self):

		self.word2idx = {'PAD' : 0}
		self.weights = []
		
		self.features = {}
		self.word_list = []

		self.glove_path = 'glove.840B.300d.txt'
		self.dataset_path_train = 'msr-paraphrase-corpus/msr_paraphrase_train.txt'
		self.dataset_path_test = 'msr-paraphrase-corpus/msr_paraphrase_test.txt'

		self.learning_rate = 0.0001


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
			
			self.weights = np.stack(self.weights)
			
			
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

		with open(self.dataset_path_train,'r') as file1:
			
			self.sentence_one = []
			self.sentence_two = []
			self.y_true = []

			for index, line in enumerate(file1):
				
				if index == 0:
					continue



				s_1 = []
				s_2 = []

				values = line.split("\t")
				
				#Sentence 1
				words = values[3].split(" ")
				
				s_1.extend([self.word2idx.get(word,self.word2idx['UNK']) for word in words])

				#Sentence 2
				words = values[4].split(" ")
				
				s_2.extend([self.word2idx.get(word,self.word2idx['UNK']) for word in words])
				


				self.y_true.append(np.asarray(values[0]))

				self.sentence_one.append(np.asarray(s_1[0:self.sen_len]))
				self.sentence_two.append(np.asarray(s_2[0:self.sen_len]))

		self.sentence_one = np.stack(self.sentence_one)
		self.sentence_two = np.stack(self.sentence_two)
		self.y_true = np.stack(self.y_true)

		print self.sentence_one.shape,self.sentence_two.shape,self.y_true.shape
	


	def load_dataset_test(self):

		with open(self.dataset_path_test,'r') as file1:
			
			self.sentence_one_test = []
			self.sentence_two_test = []
			self.y_true_test = []

			for index, line in enumerate(file1):
				
				if index == 0:
					continue

				s_1 = []
				s_2 = []

				values = line.split("\t")
				
				#Sentence 1
				words = values[3].split(" ")
				
				s_1.extend([self.word2idx.get(word,self.word2idx['UNK']) for word in words])

				#Sentence 2
				words = values[4].split(" ")
				
				s_2.extend([self.word2idx.get(word,self.word2idx['UNK']) for word in words])
				


				self.y_true_test.append(np.asarray(values[0]))

				self.sentence_one_test.append(np.asarray(s_1[0:self.sen_len]))
				self.sentence_two_test.append(np.asarray(s_2[0:self.sen_len]))

		self.sentence_one_test = np.stack(self.sentence_one_test)
		self.sentence_two_test = np.stack(self.sentence_two_test)
		self.y_true_test = np.stack(self.y_true_test)

		#print self.weights
		print self.sentence_one_test.shape,self.sentence_two_test.shape,self.y_true_test.shape


if __name__ == '__main__':
	
	x = model()

	x.sen_len = 5

	x.load_glove()
	x.load_dataset()

	x.load_dataset_test()

	dropout_rate = 0.1
	num_epoch = 10
	batch_size = 522
	
 
	X_init = tf.placeholder(tf.float32, shape=(x.vocab_size,x.embed_dim))

	sentence_one = tf.placeholder("int32",[batch_size, x.sen_len],name="sentence_one")
	sentence_two = tf.placeholder("int32",[batch_size, x.sen_len],name="sentence_two")

	Y = tf.placeholder("int32",[batch_size, 2],name="true_labels")

	embedding_weights = tf.get_variable(
		name = 'embedding_weights',
		initializer = X_init,
		trainable = False)

	# sentence_one_len = tf.shape(sentence_one)[1]
	# sentence_two_len = tf.shape(sentence_two)[1]

	# print sentence_one_len

	with tf.name_scope("Word_embeddings"):

		embedded_sentence_one = tf.nn.embedding_lookup(embedding_weights,sentence_one)
		embedded_sentence_two = tf.nn.embedding_lookup(embedding_weights,sentence_two)

	# print embedded_sentence_one[0].get_shape()
	# print embedded_sentence_two.get_shape()

	with tf.variable_scope("Context_representation_layer"):

		context_rnn_hidden_size = 100

		sentence_enc_fw = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)
		sentence_enc_bw = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)

		# Sentence 1

		outputs_1, states_1  = tf.nn.bidirectional_dynamic_rnn(
		    cell_fw=sentence_enc_fw,
		    cell_bw=sentence_enc_bw,
		    dtype=tf.float32,
		    # sequence_length=sentence_one_len,
		    inputs=embedded_sentence_one)

		output_fw_1, output_bw_1 = outputs_1
		states_fw_1, states_bw_1 = states_1

		output_fw_1 = tf.layers.dropout(
			output_fw_1,
			rate=dropout_rate,
			training=False,
			name="sentence_1_fw_dropout")
		output_bw_1 = tf.layers.dropout(
			output_bw_1,
			rate=dropout_rate,
			training=False,
			name="sentence_1_bw_dropout")

		tf.get_variable_scope().reuse_variables()

		#Sentence 2

		outputs_2, states_2  = tf.nn.bidirectional_dynamic_rnn(
		    cell_fw=sentence_enc_fw,
		    cell_bw=sentence_enc_bw,
		    dtype=tf.float32,
		    # sequence_length=sentence_two_len,
		    inputs=embedded_sentence_two)
		 
		output_fw_2, output_bw_2 = outputs_2
		states_fw_2, states_bw_2 = states_2

		output_fw_2 = tf.layers.dropout(
			output_fw_2,
			rate=dropout_rate,
			training=False,
			name="sentence_2_fw_dropout")
		output_bw_2 = tf.layers.dropout(
			output_bw_2,
			rate=dropout_rate,
			training=False,
			name="sentence_2_bw_dropout")


	#tf.get_variable_scope().reuse_variables()
	with tf.variable_scope("Matching_layer"):
		
		match_1_2,match_2_1 = bidirectional_match(output_fw_1,output_bw_1,output_fw_2,output_bw_2)

		match_1_2 = tf.layers.dropout(
			match_1_2,
			rate=dropout_rate,
			training=False,
			name="match_1_2_dropout")
		match_2_1 = tf.layers.dropout(
			match_2_1,
			rate=dropout_rate,
			training=False,
			name="match_2_1_dropout")


	## Aggregation Layer

	with tf.variable_scope("Aggregation_layer"):

		agg_last = []

		with tf.variable_scope("agg_1",reuse=tf.AUTO_REUSE):

			aggregation_enc_fw = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)
			aggregation_enc_bw = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)

			# Sentence 1

			agg_outputs_1, agg_states_1  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=aggregation_enc_fw,
			    cell_bw=aggregation_enc_bw,
			    dtype=tf.float32,
			    #sequence_length=[sentence_one_len],
			    inputs=match_1_2)

			agg_output_fw_1, agg_output_bw_1 = agg_outputs_1

			agg_output_fw_1 = tf.layers.dropout(
				agg_output_fw_1,
				rate=dropout_rate,
				training=False,
				name="agg_1_fw_dropout")
			agg_output_bw_1 = tf.layers.dropout(
				agg_output_bw_1,
				rate=dropout_rate,
				training=False,
				name="agg_1_bw_dropout")



			agg_output_fw_1_last = get_last_output(agg_output_fw_1)
			agg_output_bw_1_last = get_last_output(agg_output_bw_1)

			agg_last.append(agg_output_fw_1_last)
			agg_last.append(agg_output_bw_1_last)

		#tf.get_variable_scope().reuse_variables()

		#Sentence 2

		with tf.variable_scope("agg_2",reuse=tf.AUTO_REUSE):

			aggregation_enc_fw_2 = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)
			aggregation_enc_bw_2 = tf.nn.rnn_cell.LSTMCell(context_rnn_hidden_size,state_is_tuple=True)

			agg_outputs_2, agg_states_2  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=aggregation_enc_fw_2,
			    cell_bw=aggregation_enc_bw_2,
			    dtype=tf.float32,
			    #sequence_length=[sentence_two_len],
			    inputs=match_2_1)
			 
			agg_output_fw_2, agg_output_bw_2 = agg_outputs_2

			agg_output_fw_2 = tf.layers.dropout(
				agg_output_fw_2,
				rate=dropout_rate,
				training=False,
				name="agg_2_fw_dropout")
			agg_output_bw_2 = tf.layers.dropout(
				agg_output_bw_2,
				rate=dropout_rate,
				training=False,
				name="agg_2_bw_dropout")



			agg_output_fw_2_last = get_last_output(agg_output_fw_2)
			agg_output_bw_2_last = get_last_output(agg_output_bw_2)


			agg_last.append(agg_output_fw_2_last)
			agg_last.append(agg_output_bw_2_last)

		
		combined_agg_last = tf.concat(agg_last,1)


	##Prediction Layer
	with tf.variable_scope("Prediction_layer",reuse=tf.AUTO_REUSE):

		prediction_layer_1 = tf.layers.dense(combined_agg_last,
			combined_agg_last.get_shape().as_list()[1],
			activation=tf.nn.tanh,
			name="pred_1")
		
		prediction_layer_1 = tf.layers.dropout(
			prediction_layer_1,
			rate=dropout_rate,
			training=False,
			name="pred_dropout")



		prediction_layer_2 = tf.layers.dense(prediction_layer_1,2,name="pred_2")

	with tf.variable_scope("Train_loss_and_acc"):

		y_pred = tf.nn.softmax(prediction_layer_2)
		loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=prediction_layer_2))

		correct_predictions = tf.equal(tf.argmax(y_pred, 1),tf.argmax(Y, 1))

		batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"))

		optimizer = tf.train.AdamOptimizer(learning_rate=x.learning_rate)
		train_op = optimizer.minimize(loss_op)


	saver = tf.train.Saver()


	with tf.Session() as sess:

		trained_model = os.path.join('trained_model', 'model.ckpt')
		trained_model_restore = os.path.join('trained_model', 'model.ckpt.meta')

		if os.path.isfile(trained_model_restore):

			print "Loading saved model..."
			saver.restore(sess, trained_model)

		else:

			print "No saved model found...Training..."
			sess.run(tf.global_variables_initializer(),feed_dict={X_init:x.weights})

			i = 0.0
			sum_acc = 0.0

			for epoch in range(num_epoch):

				for step in range(x.sentence_one.shape[0]/batch_size):

					i += 1

					batch_s1,batch_s2,batch_y = x.sentence_one[step*batch_size:(step+1)*batch_size],\
												x.sentence_two[step*batch_size:(step+1)*batch_size],\
												x.y_true[step*batch_size:(step+1)*batch_size]

					#print batch_s1.shape,batch_s2.shape,batch_y.shape

					batch_y = one_hot(batch_size,batch_y.astype(int))

					#print batch_y
					

					# batch_s1 = np.expand_dims(batch_s1[0][0:sen_len],1).T
					# batch_s2 = np.expand_dims(batch_s2[0][0:sen_len],1).T

					#print batch_s1.shape,batch_s2.shape

					[_,loss,acc] = sess.run([train_op,loss_op,batch_accuracy],
									feed_dict={X_init:x.weights,
									sentence_one: batch_s1,
									sentence_two: batch_s2,
									Y:batch_y
									})


					sum_acc += acc

					print "Epoch:" + str(epoch+1) + " Step:" + str(step) + " Loss:" + "{:.4f}".format(loss) + " Batch Acc:" + "{:.4f}".format(acc) + " Mean Batch Acc:" + "{:.4f}".format(sum_acc/i)
		
			save_path = saver.save(sess, trained_model)
			print "Model saved in path: %s" % save_path

		print "Testing Model..."
		i = 0
		sum_acc = 0.0
		for step in range(x.sentence_one_test.shape[0]/batch_size):

			i += 1

			batch_s1,batch_s2,batch_y = x.sentence_one_test[step*batch_size:(step+1)*batch_size],\
										x.sentence_two_test[step*batch_size:(step+1)*batch_size],\
										x.y_true_test[step*batch_size:(step+1)*batch_size]

			batch_y = one_hot(batch_size,batch_y.astype(int))

			[acc] = sess.run([batch_accuracy],
							feed_dict={X_init:x.weights,
							sentence_one: batch_s1,
							sentence_two: batch_s2,
							Y:batch_y
							})
			sum_acc += acc

			print "Batch Test Accuracy:" + "{:.4f}".format(acc) + " Mean Batch Test Acc:" + "{:.4f}".format(sum_acc/i)

