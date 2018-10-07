import tensorflow as tf
from tensorflow import keras
import collections
import io

path='dict.txt'

dictionary={}
reverse_dictionary={}
with open(path,'r',encoding='UTF-8') as f:
	for line in f:
		key=line.split()[0]
		value=line.split()[1]
		dictionary[key]=value
		reverse_dictionary[value]=key

print(dictionary['你好'])
# sess=tf.Session()

# saver=tf.train.import_meta_graph('log/model.ckpt.meta')
# saver.restore(sess,tf.train.latest_checkpoint('./log/'))



#v0=tf.initialize_variables;


print("")