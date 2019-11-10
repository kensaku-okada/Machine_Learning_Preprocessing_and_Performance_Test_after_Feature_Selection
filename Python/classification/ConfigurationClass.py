#!/usr/bin/python3
#coding:utf-8

import Constant

class ConfigurationClass:

	# constructor
	def __init__(self):
		# output dataset form type: densed or sparsed
		self._outputDatasetType = None
		# output dataset form type: currently only "arff" is accespted
		self._importDatasetFormat = None
		# https://newtechnologylifestyle.net/%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%80%81%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E3%81%A7%E3%81%AE%E5%AD%A6%E7%BF%92%E3%83%87%E3%83%BC%E3%82%BF%E3%81%A8/
		self._crossValidationType = None
		#########################
		# if you want to process multiple files, make it True
		# if you want to process only one file, make it False
		self._ifGetMultipleResults = None
		#########################
		self._test_results = None
		# compared classifiers
		self._classifier_names = None
		self._dataset_name = None
		self._dataset_directory = None

	@property
	def outputDatasetType(self):
		return self._outputDatasetType
	@outputDatasetType.setter
	def outputDatasetType(self, outputDatasetType):
		self._outputDatasetType = outputDatasetType

	@property
	def importDatasetFormat(self):
		return self._importDatasetFormat
	@importDatasetFormat.setter
	def importDatasetFormat(self, importDatasetFormat):
		self._importDatasetFormat = importDatasetFormat

	@property
	def crossValidationType(self):
		return self._crossValidationType
	@crossValidationType.setter
	def crossValidationType(self, crossValidationType):
		self._crossValidationType = crossValidationType

	@property
	def ifGetMultipleResults(self):
		return self._ifGetMultipleResults
	@ifGetMultipleResults.setter
	def ifGetMultipleResults(self, ifGetMultipleResults):
		self._ifGetMultipleResults = ifGetMultipleResults

	@property
	def test_results(self):
		return self._test_results
	@test_results.setter
	def test_results(self, test_results):
		self._test_results = test_results

	@property
	def classifier_names(self):
		return self._classifier_names
	@classifier_names.setter
	def classifier_names(self, classifier_names):
		self._classifier_names = classifier_names

	@property
	def dataset_name(self):
		return self._dataset_name
	@dataset_name.setter
	def dataset_name(self, dataset_name):
		self._dataset_name = dataset_name

	@property
	def dataset_directory(self):
		return self._dataset_directory
	@dataset_directory.setter
	def dataset_directory(self, dataset_directory):
		self._dataset_directory = dataset_directory
