'''
Train the model on dataset
'''
import tensorflow as tf
from settings import *
from model import SSDModel
from model import ModelHelper
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import math
import os
import time
import pickle
from PIL import Image


def next_batch(X, y_conf, y_loc, batch_size):
	"""
	Next batch generator
	Arguments:
		* X: List of image file names
		* y_conf: List of ground-truth vectors for class labels
		* y_loc: List of ground-truth vectors for localization
		* batch_size: Batch size

	Yields:
		* images: Batch numpy array representation of batch of images
		* y_true_conf: Batch numpy array of ground-truth class labels
		* y_true_loc: Batch numpy array of ground-truth localization
		* conf_loss_mask: Loss mask for confidence loss, to set NEG_POS_RATIO
	"""
	start_idx = 0
	while True:
		image_files = X[start_idx : start_idx + batch_size]
		y_true_conf = np.array(y_conf[start_idx : start_idx + batch_size])
		y_true_loc  = np.array(y_loc[start_idx : start_idx + batch_size])

		# Read images from image_files
		images = []
		for image_file in image_files:
			image = Image.open('resized_images_%sx%s/%s' % (IMG_W, IMG_H, image_file))
			image = np.asarray(image)
			images.append(image)

		images = np.array(images, dtype='float32')

		# Grayscale images have array shape (H, W), but we want shape (H, W, 1)
		if NUM_CHANNELS == 1:
			images = np.expand_dims(images, axis=-1)

		# Normalize pixel values (scale them between -1 and 1)
		images = images/127.5 - 1.

		# For y_true_conf, calculate how many negative examples we need to satisfy NEG_POS_RATIO
		num_pos = np.where(y_true_conf > 0)[0].shape[0]
		num_neg = NEG_POS_RATIO * num_pos
		y_true_conf_size = np.sum(y_true_conf.shape)

		# Create confidence loss mask to satisfy NEG_POS_RATIO
		if num_pos + num_neg < y_true_conf_size:
			conf_loss_mask = np.copy(y_true_conf)
			conf_loss_mask[np.where(conf_loss_mask > 0)] = 1.

			# Find all (i,j) tuples where y_true_conf[i][j]==0
			zero_indices = np.where(conf_loss_mask == 0.)  # ([i1, i2, ...], [j1, j2, ...])
			zero_indices = np.transpose(zero_indices)  # [[i1, j1], [i2, j2], ...]

			# Randomly choose num_neg rows from zero_indices, w/o replacement
			chosen_zero_indices = zero_indices[np.random.choice(zero_indices.shape[0], int(num_neg), False)]

			# "Enable" chosen negative examples, specified by chosen_zero_indices
			for zero_idx in chosen_zero_indices:
				i, j = zero_idx
				conf_loss_mask[i][j] = 1.

		else:
			# If we have so many positive examples such that num_pos+num_neg >= y_true_conf_size,
			# no need to prune negative data
			conf_loss_mask = np.ones_like(y_true_conf)

		yield (images, y_true_conf, y_true_loc, conf_loss_mask)

		# Update start index for the next batch
		start_idx += batch_size
		if start_idx >= X.shape[0]:
			start_idx = 0


def run_training():
	"""
	Load training and test data
	Run training process
	Plot train/validation losses
	Report test loss
	Save model
	"""
	# Load training and test data
	with open('data_prep_%sx%s.p' % (IMG_W, IMG_H), mode='rb') as f:
		train = pickle.load(f)
	#with open('test.p', mode='rb') as f:
	#	test = pickle.load(f)

	# Format the data
	X_train = []
	y_train_conf = []
	y_train_loc = []
	for image_file in train.keys():
		X_train.append(image_file)
		y_train_conf.append(train[image_file]['y_true_conf'])
		y_train_loc.append(train[image_file]['y_true_loc'])
	X_train = np.array(X_train)
	y_train_conf = np.array(y_train_conf)
	y_train_loc = np.array(y_train_loc)

	# Train/validation split
	X_train, X_valid, y_train_conf, y_valid_conf, y_train_loc, y_valid_loc = train_test_split(\
		X_train, y_train_conf, y_train_loc, test_size=VALIDATION_SIZE, random_state=1)

	# Launch the graph
	with tf.Graph().as_default(), tf.Session() as sess:
		# "Instantiate" neural network, get relevant tensors
		model = SSDModel()
		x = model['x']
		y_true_conf = model['y_true_conf']
		y_true_loc = model['y_true_loc']
		conf_loss_mask = model['conf_loss_mask']
		is_training = model['is_training']
		optimizer = model['optimizer']
		reported_loss = model['loss']

		# Training process
		# TF saver to save/restore trained model
		saver = tf.train.Saver()

		if RESUME:
			print('Restoring previously trained model at %s' % MODEL_SAVE_PATH)
			saver.restore(sess, MODEL_SAVE_PATH)

			# Restore previous loss history
			with open('loss_history.p', 'rb') as f:
				loss_history = pickle.load(f)
		else:
			print('Training model from scratch')
			# Variable initialization
			sess.run(tf.global_variables_initializer())

			# For book-keeping, keep track of training and validation loss over epochs, like such:
			# [(train_acc_epoch1, valid_acc_epoch1), (train_acc_epoch2, valid_acc_epoch2), ...]
			loss_history = []

		# Record time elapsed for performance check
		last_time = time.time()
		train_start_time = time.time()

		# Run NUM_EPOCH epochs of training
		for epoch in range(NUM_EPOCH):
			train_gen = next_batch(X_train, y_train_conf, y_train_loc, BATCH_SIZE)
			num_batches_train = math.ceil(X_train.shape[0] / BATCH_SIZE)
			losses = []  # list of loss values for book-keeping

			# Run training on each batch
			for _ in range(num_batches_train):
				# Obtain the training data and labels from generator
				images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(train_gen)

				# Perform gradient update (i.e. training step) on current batch
				_, loss = sess.run([optimizer, reported_loss], feed_dict={
				#_, loss, loc_loss_dbg, loc_loss_mask, loc_loss = sess.run([optimizer, reported_loss, model['loc_loss_dbg'], model['loc_loss_mask'], model['loc_loss']],feed_dict={  # DEBUG
					x: images,
					y_true_conf: y_true_conf_gen,
					y_true_loc: y_true_loc_gen,
					conf_loss_mask: conf_loss_mask_gen,
					is_training: True
				})
				
				losses.append(loss)  # TODO: Need mAP metric instead of raw loss

			# A rough estimate of loss for this epoch (overweights the last batch)
			train_loss = np.mean(losses)

			# Calculate validation loss at the end of the epoch
			valid_gen = next_batch(X_valid, y_valid_conf, y_valid_loc, BATCH_SIZE)
			num_batches_valid = math.ceil(X_valid.shape[0] / BATCH_SIZE)
			losses = []
			for _ in range(num_batches_valid):
				images, y_true_conf_gen, y_true_loc_gen, conf_loss_mask_gen = next(valid_gen)

				# Perform forward pass and calculate loss
				loss = sess.run(reported_loss, feed_dict={
					x: images,
					y_true_conf: y_true_conf_gen,
					y_true_loc: y_true_loc_gen,
					conf_loss_mask: conf_loss_mask_gen,
					is_training: False
				})
				losses.append(loss)
			valid_loss = np.mean(losses)

			# Record and report train/validation/test losses for this epoch
			loss_history.append((train_loss, valid_loss))

			# Print accuracy every epoch
			print('Epoch %d -- Train loss: %.4f, Validation loss: %.4f, Elapsed time: %.2f sec' %\
				(epoch+1, train_loss, valid_loss, time.time() - last_time))
			last_time = time.time()

		total_time = time.time() - train_start_time
		print('Total elapsed time: %d min %d sec' % (total_time/60, total_time%60))

		test_loss = 0.  # TODO: Add test set
		'''
		# After training is complete, evaluate accuracy on test set
		print('Calculating test accuracy...')
		test_gen = next_batch(X_test, y_test, BATCH_SIZE)
		test_size = X_test.shape[0]
		test_acc = calculate_accuracy(test_gen, test_size, BATCH_SIZE, accuracy, x, y, keep_prob, sess)
		print('Test acc.: %.4f' % (test_acc,))
		'''

		if SAVE_MODEL:
			# Save model to disk
			save_path = saver.save(sess, MODEL_SAVE_PATH)
			print('Trained model saved at: %s' % save_path)

			# Also save accuracy history
			print('Loss history saved at loss_history.p')
			with open('loss_history.p', 'wb') as f:
				pickle.dump(loss_history, f)

	# Return final test accuracy and accuracy_history
	return test_loss, loss_history


if __name__ == '__main__':
	run_training()
