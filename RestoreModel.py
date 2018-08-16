import tensorflow as tf


sess=tf.Session()

saver=tf.train.import_meta_graph('log\\model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./log/'))

v1=tf.get_variable("")