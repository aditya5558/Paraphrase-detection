import tensorflow as tf

eps = 1e-6

def cosine_sim(x1,x2):

	# x1 shape : (batch_size, 1, sequence_length, rnn_hidden_size)
	# x2 shape : (batch_size, sequence_length, 1, rnn_hidden_size)

	# Shape : (batch_size, sequence_length, sequence_length)
	numerator = tf.reduce_sum(tf.multiply(x1,x2),axis=-1)

	# Shape : (batch_size, 1, sequence_length)
	x1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x1),axis=-1),eps))

	# Shape : (batch_size, sequence_length, 1)
	x2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x2),axis=-1),eps))

	return numerator / (x1_norm * x2_norm)


def cosine_matrix(x1,x2):

	# x1 shape : (batch_size, sequence_length, rnn_hidden_size)
	# x2 shape : (batch_size, sequence_length, rnn_hidden_size)

	# Shape : (batch_size, 1, sequence_length, rnn_hidden_size)
	x1_exp = tf.expand_dims(x1,1)

	# Shape : (batch_size, sequence_length, 1, rnn_hidden_size)
	x2_exp = tf.expand_dims(x2,2)

	# Shape : (batch_size, sequence_length, sequence_length)
	cosine_sim_matrix = cosine_sim(x1_exp,x2_exp)

	return cosine_sim_matrix

def get_last_output(output):

	#print tf.shape(output)[1]
	
	output = tf.transpose(output, [1, 0, 2])
	last = tf.gather(output, int(output.get_shape()[0]) - 1)
	
	#last = tf.gather(output, int(tf.shape(output)[0]) - 1)

	# Shape : (batch_size, 1, rnn_hidden_size) 
	return last


def expand_1D(t,weights):

	# Shape : (1, rnn_hidden_size)
	t_exp = tf.expand_dims(t,axis=0)

	# Shape : (perspectives, rnn_hidden_size)
	return tf.multiply(t_exp,weights)

def expand_2D(t,weights):

	# Shape : (sequence_length, 1, rnn_hidden_size)
	t_exp = tf.expand_dims(t,axis=1)

	# Shape : (1, perspectives, rnn_hidden_size)
	weights_exp = tf.expand_dims(weights,axis=0)

	# Shape : (sequence_length, perspectives, rnn_hidden_size)
	return tf.multiply(t_exp,weights_exp)


def full_matching(sentence_a_full,sentence_b_last,weights):

	def single_func(input):

		# Shape : (sequence_length, rnn_hidden_size)
		sentence_a_single = input[0]

		# Shape : (rnn_hidden_size)
		sentence_b_last_single = input[1]

		# Shape : (sequence_length, perspectives, rnn_hidden_size)
		sentence_a_single = expand_2D(sentence_a_single,weights)

		# Shape : (perspectives, rnn_hidden_size)
		sentence_b_last_single = expand_1D(sentence_b_last_single,weights)

		# Shape : (1, perspectives, rnn_hidden_size)
 		sentence_b_last_single = tf.expand_dims(sentence_b_last_single,0)

 		# Shape : (sequence_length, perspectives)
		return cosine_sim(sentence_a_single,sentence_b_last_single)

	# Shape : (batch_size, sequence_length, perspectives)
	return tf.map_fn(single_func,(sentence_a_full,sentence_b_last),dtype="float")



def pooling_matching(sentence_a_full,sentence_b_full,weights):

	def single_func(input):

		# Shape : (sequence_length, rnn_hidden_size)
		sentence_a_single = input[0]

		# Shape : (sequence_length, rnn_hidden_size)
		sentence_b_single = input[1]

		# Shape : (sequence_length, perspectives, rnn_hidden_size)
		sentence_a_single = expand_2D(sentence_a_single,weights)			
		sentence_b_single = expand_2D(sentence_b_single,weights)

		# Shape : (sequence_length, 1, perspectives, rnn_hidden_size)
		sentence_a_single = tf.expand_dims(sentence_a_single,1)

		# Shape : (1, sequence_length, perspectives, rnn_hidden_size)
		sentence_b_single = tf.expand_dims(sentence_b_single,0)

		# Shape : (sequence_length, sequence_length, perspectives)
		return cosine_sim(sentence_a_single,sentence_b_single)

	# Shape : (batch_size, sequence_length, sequence_length, perspectives)
	match_matrix = tf.map_fn(single_func,(sentence_a_full,sentence_b_full),dtype="float")

	# Shape : (batch_size, sequence_length, perspectives)
	return tf.reduce_max(match_matrix,axis=2)


def weighted_sim(sentence,sim_mat):

	# Shape : (batch_size, sequence_length, sequence_length, 1)
	sim_mat_exp = tf.expand_dims(sim_mat,-1)

	# Shape : (batch_size, 1, sequence_length, rnn_hidden_size)
	sentence = tf.expand_dims(sentence,1)

	# Shape : (batch_size, sequence_length, rnn_hidden_size)
	weighted_sim_mat = tf.reduce_sum(tf.multiply(sentence,sim_mat_exp),axis=1)

	weighted_sim_mat = tf.div(weighted_sim_mat,tf.expand_dims(tf.add(tf.reduce_sum(sim_mat,axis=-1),eps),axis=-1))

	return weighted_sim_mat


def max_sim(sentence_b_full,sim_mat):

	def single_func(input):

		# Shape : (sequence_length, rnn_hidden_size)
		sentence_b_single = input[0]

		# Shape : (sequence_length)
		max_index = input[1]

		# Shape : (sequence_length, rnn_hidden_size)
		return tf.gather(sentence_b_single,max_index)

	max_index = tf.arg_max(sim_mat,2)

	# Shape : (batch_size, sequence_length, rnn_hidden_size)
	return tf.map_fn(single_func,(sentence_b_full,max_index),dtype="float")


def attentive_matching(sentence_a_full,weighted_sim_mat,weights):

	def single_func(input):

		# Shape : (sequence_length, rnn_hidden_size)
		sentence_a_single = input[0]

		# Shape : (sequence_length, rnn_hidden_size)
		sentence_b_single_att = input[1]

		# Shape : (sequence_length, perspectives, rnn_hidden_size)
		sentence_a_single = expand_2D(sentence_a_single,weights)
		sentence_b_single_att = expand_2D(sentence_b_single_att,weights)

		# Shape : (sequence_length, perspectives)
		return cosine_sim(sentence_a_single,sentence_b_single_att)

	# Shape : (batch_size, sequence_length, perspectives)
	return tf.map_fn(single_func,(sentence_a_full,weighted_sim_mat),dtype="float")


def match(sentence_a_fw, sentence_a_bw, sentence_b_fw, sentence_b_bw, perspectives,hidden_size):

	fw_sim_matrix = cosine_matrix(sentence_a_fw,sentence_b_fw)
	bw_sim_matrix = cosine_matrix(sentence_a_bw,sentence_b_bw)

	#sentence_b_len = tf.reduce_sum(sentence_b_fw, 1)



	# tf.get_variable_scope().reuse_variables()
	with tf.variable_scope("Full_matching",reuse=tf.AUTO_REUSE): 


		last_output_b_fw = get_last_output(sentence_b_fw)
		
		weights_fw = tf.get_variable("forward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")

		fw_full_match = full_matching(sentence_a_fw,last_output_b_fw,weights_fw)

		last_output_b_bw = get_last_output(sentence_b_bw)

		weights_bw = tf.get_variable("backward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")
		
		#tf.get_variable_scope().reuse_variables()

		bw_full_match = full_matching(sentence_a_bw,last_output_b_bw,weights_bw)


	with tf.variable_scope("Pooling_matching",reuse=tf.AUTO_REUSE): 

		
		weights_fw = tf.get_variable("forward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")

		fw_pool_match = pooling_matching(sentence_a_fw,sentence_b_fw,weights_fw)

		weights_bw = tf.get_variable("backward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")
		
		#tf.get_variable_scope().reuse_variables()

		bw_pool_match = pooling_matching(sentence_a_bw,sentence_b_bw,weights_bw)

	with tf.variable_scope("Attentive_matching",reuse=tf.AUTO_REUSE): 


		sentence_b_fw_att = weighted_sim(sentence_b_fw,fw_sim_matrix)


		weights_fw = tf.get_variable("forward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")

		fw_att_match = attentive_matching(sentence_a_fw,sentence_b_fw_att,weights_fw)


		sentence_b_bw_att = weighted_sim(sentence_b_bw,bw_sim_matrix)

		weights_bw = tf.get_variable("backward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")
		
		#tf.get_variable_scope().reuse_variables()

		bw_att_match = attentive_matching(sentence_a_bw,sentence_b_bw_att,weights_bw)

	with tf.variable_scope("Max_Attentive_matching",reuse=tf.AUTO_REUSE): 


		sentence_b_fw_att_max = max_sim(sentence_b_fw,fw_sim_matrix)


		weights_fw = tf.get_variable("forward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")

		fw_max_att_match = attentive_matching(sentence_a_fw,sentence_b_fw_att_max,weights_fw)


		sentence_b_bw_att_max = max_sim(sentence_b_bw,bw_sim_matrix)

		weights_bw = tf.get_variable("backward_matching",
			shape=[perspectives,hidden_size],
			dtype="float")
		
		#tf.get_variable_scope().reuse_variables()

		bw_max_att_match = attentive_matching(sentence_a_bw,sentence_b_bw_att_max,weights_bw)

	return [fw_full_match,bw_full_match,fw_pool_match,bw_pool_match,fw_att_match,bw_att_match,fw_max_att_match,bw_max_att_match]


def bidirectional_match(sentence_a_fw,sentence_a_bw,sentence_b_fw,sentence_b_bw,perspectives=20,hidden_size=100):


	match_1_2 = match(sentence_a_fw,sentence_a_bw,sentence_b_fw,sentence_b_bw,perspectives,hidden_size)
	match_2_1 = match(sentence_b_fw,sentence_b_bw,sentence_a_fw,sentence_a_bw,perspectives,hidden_size)

	match_1_2 = tf.concat(match_1_2,2)
	match_2_1 = tf.concat(match_2_1,2)

	return match_1_2,match_2_1
