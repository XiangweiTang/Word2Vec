import tensorflow as tf


sess=tf.Session()

saver=tf.train.import_meta_graph('log\\model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('./log/'))

#v0=tf.initialize_variables;

v=tf.global_variables(scope="weight")

v1=tf.get_variable("Variable:0",shape=[50000,128])
print("")