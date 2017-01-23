'''
Visualize the model using TensorBoard
'''
import tensorflow as tf
from settings import *
from model import SSDModel

FM_ONLY = False  # Only want to see feature map sizes?

with tf.Graph().as_default(), tf.Session() as sess:
	if FM_ONLY:
		# Only want to see feature map sizes (e.g. loss function and vector concatenation not yet set up)
		if MODEL == 'AlexNet':
			from model import AlexNet as MyModel
		else:
			raise NotImplementedError('Model %s not supported' % MODEL)
		_ = MyModel()
	else:
		# This includes the entire graph, e.g. loss function, optimizer, etc.
		_ = SSDModel()

	tf.summary.merge_all()
	writer = tf.summary.FileWriter('./tensorboard_out', sess.graph)
	tf.global_variables_initializer().run()
