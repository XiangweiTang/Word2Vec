import tensorflow as tf

export_dir=".\\log\\model.ckpt.meta"
check_dir=".\\log"

with tf.Session() as sess:
	saver=tf.train.import_meta_graph(export_dir)
	saver.restore(sess,export_dir)