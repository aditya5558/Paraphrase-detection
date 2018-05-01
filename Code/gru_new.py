import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cPickle as cPickle
from matching import bidirectional_match,get_last_output
from data_loader import *

def log(message,file_path=os.path.join('glove_gru','log.txt')):

    print message
    f1=open(file_path, 'a+')
    f1.write(message)
    f1.write("\n")
    f1.close()

def one_hot(batch_size,Y):

    B = np.zeros((batch_size,2))

    B[np.arange(batch_size),Y] = 1

    return B.astype(int)

def length(sequence):
	
	used = tf.sign(sequence)
	length = tf.reduce_sum(used, 1)
	length = tf.cast(length, tf.int32)
	
	return length

class model:


	def __init__(self):

		self.word2idx = {'PAD' : 0}
		self.weights = []
		
		self.features = {}
		self.word_list = []

		self.glove_path = 'glove.840B.300d.txt'
		#self.glove_path = 'word2vec.txt'
		#self.dataset_path_train = 'paraphrase_data.tsv'
		#self.dataset_path_test = 'paraphrase_data_test.tsv'
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

					if index == 1000000:
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


if __name__ == '__main__':
	

	#os.environ['CUDA_VISIBLE_DEVICES'] = ''

	x = model()

	x.sen_len = 41

	x.load_glove()
	

	x.sentence_one,x.sentence_two,x.y_true = load_dataset(x.dataset_path_train,x.word2idx)

	x.sentence_one_test,x.sentence_two_test,x.y_true_test = load_dataset_test(x.dataset_path_test,x.word2idx)

	dropout_rate = 0.1
	num_epoch = 20
	batch_size = 256
	
 
	X_init = tf.placeholder(tf.float32, shape=(x.vocab_size,x.embed_dim))

	sentence_one = tf.placeholder("int32",[batch_size, x.sen_len],name="sentence_one")
	sentence_two = tf.placeholder("int32",[batch_size, x.sen_len],name="sentence_two")

	Y = tf.placeholder("int32",[batch_size, 2],name="true_labels")

	embedding_weights = tf.get_variable(
		name = 'embedding_weights',
		initializer = X_init,
		trainable = False)


	# print sentence_one_len

	with tf.name_scope("Word_embeddings"):

		embedded_sentence_one = tf.nn.embedding_lookup(embedding_weights,sentence_one)
		embedded_sentence_two = tf.nn.embedding_lookup(embedding_weights,sentence_two)


	with tf.variable_scope("Context_representation_layer"):

		context_rnn_hidden_size = 100

		sentence_enc_fw = tf.nn.rnn_cell.GRUCell(context_rnn_hidden_size)
		sentence_enc_bw = tf.nn.rnn_cell.GRUCell(context_rnn_hidden_size)

		# Sentence 1
		# Shape (batch_size, sequence_length, rnn_hidden_size)

		outputs_1, states_1  = tf.nn.bidirectional_dynamic_rnn(
		    cell_fw=sentence_enc_fw,
		    cell_bw=sentence_enc_bw,
		    dtype=tf.float32,
		    #sequence_length=length(sentence_one),
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

		# Shape (batch_size, sequence_length, rnn_hidden_size)

		outputs_2, states_2  = tf.nn.bidirectional_dynamic_rnn(
		    cell_fw=sentence_enc_fw,
		    cell_bw=sentence_enc_bw,
		    dtype=tf.float32,
		    #sequence_length=length(sentence_two),
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
		
		# Shape (batch_size, sequence_length, 8*perspectives)


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

		# Sentence 1

		with tf.variable_scope("agg_1",reuse=tf.AUTO_REUSE):

			aggregation_enc_fw = tf.nn.rnn_cell.GRUCell(context_rnn_hidden_size)
			aggregation_enc_bw = tf.nn.rnn_cell.GRUCell(context_rnn_hidden_size)	

			# Shape (batch_size, sequence_length, rnn_hidden_size)


			agg_outputs_1, agg_states_1  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=aggregation_enc_fw,
			    cell_bw=aggregation_enc_bw,
			    dtype=tf.float32,
			   # sequence_length=length(sentence_one),
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

			# Shape : (batch_size, rnn_hidden_size)

			agg_output_fw_1_last = get_last_output(agg_output_fw_1)
			agg_output_bw_1_last = get_last_output(agg_output_bw_1)

			agg_last.append(agg_output_fw_1_last)
			agg_last.append(agg_output_bw_1_last)

		#tf.get_variable_scope().reuse_variables()

		#Sentence 2

		with tf.variable_scope("agg_2",reuse=tf.AUTO_REUSE):


			aggregation_enc_fw_2 = tf.nn.rnn_cell.GRUCell(context_rnn_hidden_size)
			aggregation_enc_bw_2 = tf.nn.rnn_cell.GRUCell(context_rnn_hidden_size)

			# Shape (batch_size, sequence_length, rnn_hidden_size)

			agg_outputs_2, agg_states_2  = tf.nn.bidirectional_dynamic_rnn(
			    cell_fw=aggregation_enc_fw_2,
			    cell_bw=aggregation_enc_bw_2,
			    dtype=tf.float32,
			    #sequence_length=length(sentence_two),
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


			# Shape : (batch_size, rnn_hidden_size)

			agg_output_fw_2_last = get_last_output(agg_output_fw_2)
			agg_output_bw_2_last = get_last_output(agg_output_bw_2)


			agg_last.append(agg_output_fw_2_last)
			agg_last.append(agg_output_bw_2_last)

		# Shape : (batch_size, 4*rnn_hidden_size)

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
		
		tf.summary.scalar('loss',loss_op)		

		correct_predictions = tf.equal(tf.argmax(y_pred, 1),tf.argmax(Y, 1))

		batch_accuracy = tf.reduce_mean(tf.cast(correct_predictions,"float"))
		tf.summary.scalar('accuracy',batch_accuracy)

		optimizer = tf.train.AdamOptimizer(learning_rate=x.learning_rate)
		train_op = optimizer.minimize(loss_op)

	merged = tf.summary.merge_all()
	saver = tf.train.Saver()


	with tf.Session() as sess:

		trained_model = os.path.join('glove_gru', 'model.ckpt')
		trained_model_restore = os.path.join('glove_gru', 'model.ckpt.meta')

		if os.path.isfile(trained_model_restore):

			print "Loading saved model..."
			saver.restore(sess, trained_model)

		else:
			train_writer = tf.summary.FileWriter("glove_gru/",sess.graph)
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

					[summary,_,loss,acc] = sess.run([merged,train_op,loss_op,batch_accuracy],
									feed_dict={X_init:x.weights,
									sentence_one: batch_s1,
									sentence_two: batch_s2,
									Y:batch_y
									})

					train_writer.add_summary(summary, i)


					sum_acc += acc

					log("Epoch:" + str(epoch+1) + " Step:" + str(step) + " Loss:" + "{:.4f}".format(loss) + " Batch Acc:" + "{:.4f}".format(acc) + " Mean Batch Acc:" + "{:.4f}".format(sum_acc/i))
		
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

			log("Batch Test Accuracy:" + "{:.4f}".format(acc) + " Mean Batch Test Acc:" + "{:.4f}".format(sum_acc/i))

