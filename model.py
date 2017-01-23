'''
Model definition
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from settings import *
from data_prep import calc_iou


def SSDHook(feature_map, hook_id):
	"""
	Takes input feature map, output the predictions tensor
	hook_id is for variable_scope unqie string ID
	"""
	with tf.variable_scope('ssd_hook_' + hook_id):
		# Note we have linear activation (i.e. no activation function)
		net_conf = slim.conv2d(feature_map, NUM_PRED_CONF, [3, 3], activation_fn=None, scope='conv_conf')
		net_conf = tf.contrib.layers.flatten(net_conf)

		net_loc = slim.conv2d(feature_map, NUM_PRED_LOC, [3, 3], activation_fn=None, scope='conv_loc')
		net_loc = tf.contrib.layers.flatten(net_loc)

	return net_conf, net_loc


def ModelHelper(y_pred_conf, y_pred_loc):
	"""
	Define loss function, optimizer, predictions, and accuracy metric
	Loss includes confidence loss and localization loss

	conf_loss_mask is created at batch generation time, to mask the confidence losses
	It has 1 at locations w/ positives, and 1 at select negative locations
	such that negative-to-positive ratio of NEG_POS_RATIO is satisfied

	Arguments:
		* y_pred_conf: Class predictions from model,
			a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * num_classes]
		* y_pred_loc: Localization predictions from model,
			a tensor of shape [batch_size, num_feature_map_cells * num_defaul_boxes * 4]

	Returns relevant tensor references
	"""
	num_total_preds = 0
	for fm_size in FM_SIZES:
		num_total_preds += fm_size[0] * fm_size[1] * NUM_DEFAULT_BOXES
	num_total_preds_conf = num_total_preds * NUM_CLASSES
	num_total_preds_loc  = num_total_preds * 4

	# Input tensors
	y_true_conf = tf.placeholder(tf.int32, [None, num_total_preds], name='y_true_conf')  # classification ground-truth labels
	y_true_loc  = tf.placeholder(tf.float32, [None, num_total_preds_loc], name='y_true_loc')  # localization ground-truth labels
	conf_loss_mask = tf.placeholder(tf.float32, [None, num_total_preds], name='conf_loss_mask')  # 1 mask "bit" per def. box

	# Confidence loss
	logits = tf.reshape(y_pred_conf, [-1, num_total_preds, NUM_CLASSES])
	conf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y_true_conf)
	conf_loss = conf_loss_mask * conf_loss  # "zero-out" the loss for don't-care negatives
	conf_loss = tf.reduce_sum(conf_loss)

	# Localization loss (smooth L1 loss)
	# loc_loss_mask is analagous to conf_loss_mask, except 4 times the size
	diff = y_true_loc - y_pred_loc
	
	loc_loss_l2 = 0.5 * (diff**2.0)
	loc_loss_l1 = tf.abs(diff) - 0.5
	smooth_l1_condition = tf.less(tf.abs(diff), 1.0)
	loc_loss = tf.select(smooth_l1_condition, loc_loss_l2, loc_loss_l1)
	
	loc_loss_mask = tf.minimum(y_true_conf, 1)  # have non-zero localization loss only where we have matching ground-truth box
	loc_loss_mask = tf.to_float(loc_loss_mask)
	loc_loss_mask = tf.stack([loc_loss_mask] * 4, axis=2)  # [0, 1, 1] -> [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]], ...]
	loc_loss_mask = tf.reshape(loc_loss_mask, [-1, num_total_preds_loc])  # removing the inner-most dimension of above
	loc_loss = loc_loss_mask * loc_loss
	loc_loss = tf.reduce_sum(loc_loss)

	# Weighted average of confidence loss and localization loss
	# Also add regularization loss
	loss = conf_loss + LOC_LOSS_WEIGHT * loc_loss + tf.reduce_sum(slim.losses.get_regularization_losses())
	optimizer = OPT.minimize(loss)

	#reported_loss = loss #tf.reduce_sum(loss, 1)  # DEBUG

	# Class probabilities and predictions
	probs_all = tf.nn.softmax(logits)
	probs, preds_conf = tf.nn.top_k(probs_all)  # take top-1 probability, and the index is the predicted class
	probs = tf.reshape(probs, [-1, num_total_preds])
	preds_conf = tf.reshape(preds_conf, [-1, num_total_preds])

	# Return a dictionary of {tensor_name: tensor_reference}
	ret_dict = {
		'y_true_conf': y_true_conf,
		'y_true_loc': y_true_loc,
		'conf_loss_mask': conf_loss_mask,
		'optimizer': optimizer,
		'conf_loss': conf_loss,
		'loc_loss': loc_loss,
		'loss': loss,
		'probs': probs,
		'preds_conf': preds_conf,
		'preds_loc': y_pred_loc,
	}
	return ret_dict


def AlexNet():
	"""
	AlexNet
	"""
	# Image batch tensor and dropout keep prob placeholders
	x = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, NUM_CHANNELS], name='x')
	is_training = tf.placeholder(tf.bool, name='is_training')

	# Classification and localization predictions
	preds_conf = []  # conf -> classification b/c confidence loss -> classification loss
	preds_loc = []

	# Use batch normalization for all convolution layers
	# FIXME: Not sure why setting is_training is not working well
	#with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
	with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': True},\
			weights_regularizer=slim.l2_regularizer(scale=REG_SCALE)):
		net = slim.conv2d(x, 64, [11, 11], 4, padding='VALID', scope='conv1')
		net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
		net = slim.conv2d(net, 192, [5, 5], scope='conv2')

		net_conf, net_loc = SSDHook(net, 'conv2')
		preds_conf.append(net_conf)
		preds_loc.append(net_loc)

		net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
		net = slim.conv2d(net, 384, [3, 3], scope='conv3')
		net = slim.conv2d(net, 384, [3, 3], scope='conv4')
		net = slim.conv2d(net, 256, [3, 3], scope='conv5')

		# The following layers added for SSD
		net = slim.conv2d(net, 1024, [3, 3], scope='conv6')
		net = slim.conv2d(net, 1024, [1, 1], scope='conv7')

		net_conf, net_loc = SSDHook(net, 'conv7')
		preds_conf.append(net_conf)
		preds_loc.append(net_loc)

		net = slim.conv2d(net, 256, [1, 1], scope='conv8')
		net = slim.conv2d(net, 512, [3, 3], 2, scope='conv8_2')

		net_conf, net_loc = SSDHook(net, 'conv8_2')
		preds_conf.append(net_conf)
		preds_loc.append(net_loc)

		net = slim.conv2d(net, 128, [1, 1], scope='conv9')
		net = slim.conv2d(net, 256, [3, 3], 2, scope='conv9_2')

		net_conf, net_loc = SSDHook(net, 'conv9_2')
		preds_conf.append(net_conf)
		preds_loc.append(net_loc)

	# Concatenate all preds together into 1 vector, for both classification and localization predictions
	final_pred_conf = tf.concat(1, preds_conf)
	final_pred_loc = tf.concat(1, preds_loc)

	# Return a dictionary of {tensor_name: tensor_reference}
	ret_dict = {
		'x': x,
		'y_pred_conf': final_pred_conf,
		'y_pred_loc': final_pred_loc,
		'is_training': is_training,
	}
	return ret_dict


def SSDModel():
	"""
	Wrapper around the model and model helper
	Returns dict of relevant tensor references
	"""
	if MODEL == 'AlexNet':
		model = AlexNet()
	else:
		raise NotImplementedError('Model %s not supported' % MODEL)

	model_helper = ModelHelper(model['y_pred_conf'], model['y_pred_loc'])

	ssd_model = {}
	for k in model.keys():
		ssd_model[k] = model[k]
	for k in model_helper.keys():
		ssd_model[k] = model_helper[k]

	return ssd_model


def nms(y_pred_conf, y_pred_loc, prob):
	"""
	Non-Maximum Suppression (NMS)
	Performs NMS on all boxes of each class where predicted probability > CONF_THRES
	For all boxes exceeding IOU threshold, select the box with highest confidence
	Returns a lsit of box coordinates post-NMS

	Arguments:
		* y_pred_conf: Class predictions, numpy array of shape (num_feature_map_cells * num_defaul_boxes,)
		* y_pred_loc: Bounding box coordinates, numpy array of shape (num_feature_map_cells * num_defaul_boxes * 4,)
			These coordinates are normalized coordinates relative to center of feature map cell
		* prob: Class probabilities, numpy array of shape (num_feature_map_cells * num_defaul_boxes,)

	Returns:
		* boxes: Numpy array of boxes, with shape (num_boxes, 6). shape[0] is interpreted as:
			[x1, y1, x2, y2, class, probability], where x1/y1/x2/y2 are the coordinates of the
			upper-left and lower-right corners. Box coordinates assume the image size is IMG_W x IMG_H.
			Remember to rescale box coordinates if your target image has different dimensions.
	"""
	# Keep track of boxes for each class
	class_boxes = {}  # class -> [(x1, y1, x2, y2, prob), (...), ...]
	with open('signnames.csv', 'r') as f:
		for line in f:
			cls, _ = line.split(',')
			class_boxes[float(cls)] = []

	# Go through all possible boxes and perform class-based greedy NMS (greedy based on class prediction confidence)
	y_idx = 0
	for fm_size in FM_SIZES:
		fm_h, fm_w = fm_size  # feature map height and width
		for row in range(fm_h):
			for col in range(fm_w):
				for db in DEFAULT_BOXES:
					# Only perform calculations if class confidence > CONF_THRESH and not background class
					if prob[y_idx] > CONF_THRESH and y_pred_conf[y_idx] > 0.:
						# Calculate absolute coordinates of predicted bounding box
						xc, yc = col + 0.5, row + 0.5  # center of current feature map cell
						center_coords = np.array([xc, yc, xc, yc])
						abs_box_coords = center_coords + y_pred_loc[y_idx*4 : y_idx*4 + 4]  # predictions are offsets to center of fm cell

						# Calculate predicted box coordinates in actual image
						scale = np.array([IMG_W/fm_w, IMG_H/fm_h, IMG_W/fm_w, IMG_H/fm_h])
						box_coords = abs_box_coords * scale
						box_coords = [int(round(x)) for x in box_coords]

						# Compare this box to all previous boxes of this class
						cls = y_pred_conf[y_idx]
						cls_prob = prob[y_idx]
						box = (*box_coords, cls, cls_prob)
						if len(class_boxes[cls]) == 0:
							class_boxes[cls].append(box)
						else:
							suppressed = False  # did this box suppress other box(es)?
							overlapped = False  # did this box overlap with other box(es)?
							for other_box in class_boxes[cls]:
								iou = calc_iou(box[:4], other_box[:4])
								if iou > NMS_IOU_THRESH:
									overlapped = True
									# If current box has higher confidence than other box
									if box[5] > other_box[5]:
										class_boxes[cls].remove(other_box)
										suppressed = True
							if suppressed or not overlapped:
								class_boxes[cls].append(box)

					y_idx += 1

	# Gather all the pruned boxes and return them
	boxes = []
	for cls in class_boxes.keys():
		for class_box in class_boxes[cls]:
			boxes.append(class_box)
	boxes = np.array(boxes)

	return boxes
