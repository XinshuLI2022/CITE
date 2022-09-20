import tensorflow as tf

import numpy as np

sess = tf.Session()

def x_negative(x,p_score,batch_size):
	i_s = np.argsort(-p_score)
	i_1 = i_s[:batch_size]
	i_2 = i_s[(-batch_size):] 
	i_list = np.append(i_1,i_2)
	neg_list = []
	for i in range(batch_size):
		rand_num = np.random.choice(i_list,1)[0]
		neg_list.append(rand_num)
	result = x[neg_list]
	return result




def x_positive(x,p_score,batch_size):
	i_list = []
	temp = p_score.copy()
	for i in range(batch_size):
		index = (np.abs(temp-0.5)).argmin()
		i_list.append(index)
		temp[index] = 999
	i_list = np.array(i_list)
	pos_list = []
	for j in range(x.shape[0]):
		rand_num = np.random.choice(i_list,1)[0]
		pos_list.append(rand_num)

	result = x[pos_list]
	return result
