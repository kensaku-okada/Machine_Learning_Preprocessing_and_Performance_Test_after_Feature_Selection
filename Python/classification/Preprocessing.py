#!/usr/bin/python3
#coding:utf-8

import numpy as np
import Util, Constant, sys
############### package for preprocessing ######################
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
############### package for preprocessing ######################

def get_splitted_dataset_k_fold(config, dataset):
	'''
	reference source:
		https://qiita.com/KROYO/items/66d613356e2bf4cf9fed
		https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155
		https://blog.amedama.jp/entry/sklearn-cv-custom-metric
	'''
	X, y = getSplittedDataset(dataset)

	# just change the data type of y from string to int
	y = y.astype(np.int32)

	# binarize the data to resolve the following error
	X_binary = MultiLabelBinarizer().fit_transform(X)
	print("X_binary.shape: ", X_binary.shape)

	# convert from Byte code to String, and from String to int, which is necessary only when the date is byte type.
	# https://qiita.com/masakielastic/items/2a04aee632c62536f82c
	if type(y) is bytes:
		y = np.array([int(y_e.decode('utf-8')) for y_e in y])

	skf = StratifiedKFold(n_splits = Constant.NUM_FOLD_CV,
			   shuffle = True,
			   random_state = 0)

	# append is faster in the list than the array
	# https://qiita.com/ykatsu111/items/be274f76d42f6b982ba4
	# train_indices = []
	# test_indices = []
	###################################
	X_binary_train_folds = []
	y_train_folds = []
	X_binary_test_folds = []
	y_test_folds = []
	# good: http://segafreder.hatenablog.com/entry/2016/10/18/163925
	# https://ensekitt.hatenablog.com/entry/2018/08/10/200000
	# https://blog.amedama.jp/entry/2018/08/25/174530
	# https://qiita.com/chorome/items/54e99093050a9473a189
	# the followings are old:
	# 	http://segafreder.hatenablog.com/entry/2016/10/18/163925
	# folded_X_binary, folded_y = skf.split(X_binary, y)
	for train_index, test_index in skf.split(X_binary, y):
		print("train_index:", train_index, "test_index:", test_index)
		print("train_index.shape:", train_index.shape, "test_index.shape:", test_index.shape)
		# train_indices.append(train_index)
		# test_indices.append(test_index)
		X_binary_train_folds.append(X_binary[train_index])
		y_train_folds.append(y[train_index])
		X_binary_test_folds.append(X_binary[test_index])
		y_test_folds.append(y[test_index])

	# print("len(test_indices): ",len(test_indices))
	return skf, X_binary_train_folds, y_train_folds, X_binary_test_folds, y_test_folds

	# train_indices = np.array(train_indices)
	# test_indices = np.array(train_indices)
	# return skf, train_indices, test_indices


def get_splitted_dataset_hold_out(config, dataset):

	# when processing mushroom_mrmr.arff, you want to use below
	if config.dataset_name == "mushroom":
		# https://towardsdatascience.com/building-a-perfect-mushroom-classifier-ceb9d99ae87e

		# binarize the dataset
		# https://note.nkmk.me/python-pandas-get-dummies/
		dataset_binary = pd.get_dummies(dataset)
		print("type(dataset_binary): ", type(dataset_binary))
		print("dataset_binary.shape: ", dataset_binary.shape)
		print("dataset_binary.head(5):{}".format(dataset_binary.head(5)))

		X_binary, y_binary = getSplittedDataset(dataset)
		print("type(X_binary): ", type(X_binary))
		print("X_binary.shape: ", X_binary.shape)
		print("type(y_binary): ", type(y_binary))
		print("y_binary.shape: ", y_binary.shape)
		# print("y: ", y)
		# print("y[0][0]: ", y[0])
		# print("type(y[0][0]): ", type(y[0]))

		###################### old code for character binarization ######################
		# import itertools
		# # TypeError: unorderable types: NoneType() < str()
		# X_binary = np.empty((X.shape[0], X.shape[1]))
		# # source: https://www.datatechnotes.com/2019/05/one-hot-encoding-example-in-python.html
		# for i in range (0, X.shape[0]):
		#     x_numerical = LabelEncoder().fit_transform(X[i])
		#     print("x_numerical: ",x_numerical)
		#     onehot_encoder = OneHotEncoder(sparse=False)
		#     x_numerical_reshaped = x_numerical.reshape(len(x_numerical), 1)
		#     x_binary = onehot_encoder.fit_transform(x_numerical_reshaped)
		#     print("x_binary: ",x_binary)
		#     # change the 2 dim array into 1 array by concatenating the elements
		#     # https://note.nkmk.me/python-list-flatten/
		#     x_binary = list(itertools.chain.from_iterable(x_binary))
		#     X_binary[i] = x_binary
		###################### old code for character binarization ######################

		# divide data into training data and test data by 1:4 = test : train
		# if you want change the random pattern each time, remove random_state=0
		# random_state=: データを分割する際の乱数のシード値 (https://docs.pyq.jp/python/machine_learning/tips/train_test_split.html)
		# https://qiita.com/tomov3/items/039d4271ed30490edf7b
		# X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=0)
		X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=0)

	else:
		X, y = getSplittedDataset(dataset)

		# just change the data type of y from string to int
		y = y.astype(np.int32)

		# binarize the data to resolve the following error
		# ValueError: You appear to be using a legacy multi-label data representation. Sequence of sequences are no longer supported; use a binary array or sparse matrix instead.
		# source: https://stackoverflow.com/questions/34213199/scikit-learn-multilabel-classification-valueerror-you-appear-to-be-using-a-leg
		# https://github.com/phanein/deepwalk/issues/32
		X_binary = MultiLabelBinarizer().fit_transform(X)

		# divide data into training data and test data by 1:4 = test : train
		# if you want change the random pattern each time, remove random_state=0
		# random_state=: データを分割する際の乱数のシード値 (https://docs.pyq.jp/python/machine_learning/tips/train_test_split.html)
		# https://qiita.com/tomov3/items/039d4271ed30490edf7b
		# X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=0)
		X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=0)

	print("X_binary: ", X_binary)
	print("X_binary.shape: ", X_binary.shape)
	# print("y.shape: ", y.shape)
	# print("y]: ", y)
	# print("y[0]: ", y[0])
	# print("type(y[0]): ", type(y[0]))
	print("X_train.shape: ", X_train.shape)
	# print("X_train: ", X_train)
	print("X_test.shape: ", X_test.shape)
	print("y_train.shape: ", y_train.shape)
	# print("y_train: ", y_train)
	print("y_test.shape: ", y_test.shape)

	# convert from Byte code to String, and from String to int, which is necessary only when the date is byte type.
	# https://qiita.com/masakielastic/items/2a04aee632c62536f82c
	if type(y_train) is bytes:
		y_train = np.array([int(y_e.decode('utf-8')) for y_e in y_train])
	# print("y_train: ", y_train)
	# print("y_train.shape: ", y_train.shape)
	# print("type(y_train): ", type(y_train))
	# print("y_train[0]: ", y_train[0])
	# print("type(y_train[0]): ", type(y_train[0]))

	return X_train, X_test, y_train, y_test

def getSplittedDataset(dataset):
	X = dataset.iloc[:, 0:len(dataset.columns) - 1].values
	y = dataset.iloc[:, len(dataset.columns) - 1].values
	return X, y
