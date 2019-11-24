#!/usr/bin/python3
#coding:utf-8

import os, sys, datetime, glob, pathlib
import numpy as np
# import matplotlib.pyplot as plt
# from scipy.io import arff as scipy_arff # for dense arff dataset
import pandas as pd
import Util, Constant, Test, Preprocessing
import my_CrossValidation as my_CV
import ConfigurationClass as Config

############################################################
############### set the configuration start ###############
############################################################
# configuration instance
config = Config.ConfigurationClass()
config.outputDatasetType = "dense"
# config.outputDatasetType = "sparsed"
config.importDatasetFormat = "arff"

# config.crossValidationType = "hold-out"
config.crossValidationType = "k-fold-cv"

config.ifGetMultipleResults = False
config.test_results = []
# config.classifier_names = [Constant.SVC, Constant.NAIVE_BAYES, Constant.C4_5]
config.classifier_names = [Constant.SVC, Constant.NAIVE_BAYES]
# config.classifier_names = [Constant.NAIVE_BAYES]
# config.classifier_names = [Constant.SVC]
############################################################
############### set the configuration end ###############
############################################################

############################################################
############### get dataset paths start ###############
############################################################
# relativePath = "datasets\\mushroom\\mRMR\\mushroom_mrmr.arff"
relativePath = "datasets\\arcene\\mRMR\\arcene-10-disc_mrmr.arff"
# relativePath = "datasets\\dorothea\\dorothea.sparse.arff.0.5.bornfs.arff"

filePath = Util.getFilePath(relativePath)
print("filePath: ",filePath)
# https://note.nkmk.me/python-pathlib-name-suffix-parent/
dataset_directory = pathlib.Path(filePath)
print("dataset_directory: ",dataset_directory)
dataset_name = pathlib.Path(filePath).parents[1].name
print("dataset_name: ",dataset_name)
config.dataset_name = dataset_name
config.dataset_directory = dataset_directory

if config.ifGetMultipleResults:
    file_paths = glob.glob(filePath)
    # print("file_paths: ",file_paths)
else:
    # adjust the file format with the case ifGetMultipleResults = True
    file_paths = [filePath]
############################################################
############### get dataset paths end ###############
############################################################

for filePath in file_paths:

    print("filePath: ",filePath)

    # import data
    dataset = Util.importArffData(filePath, config)
    print("type(dataset): ",type(dataset))
    print("dataset.shape: ",dataset.shape)
    print("dataset.head(5):{}".format(dataset.head(5)))

    ####################################################################
    ############### preprocessing start ###############
    ####################################################################
    if config.crossValidationType == "k-fold-cv":
        # skf, train_indices, test_indices = Preprocessing.get_splitted_dataset_k_fold(config, dataset)
        X_trains, y_trains, X_tests, y_tests = Preprocessing.get_splitted_dataset_k_fold(config, dataset)
        print("X_trains: ",X_trains)
        print("y_trains: ",y_trains)
        print("X_tests: ",X_tests)
        print("y_tests: ",y_tests)

    elif config.crossValidationType == "hold-out":
        X_train, X_test, y_train, y_test = Preprocessing.get_splitted_dataset_hold_out(config, dataset)

    else:
        print("undefied cross validation type specified. stop the program")
        sys.exit(0)
    ####################################################################
    ############### preprocessing end ###############
    ####################################################################

    for classifier_name in config.classifier_names:

        clssifier_type, parameters = my_CV.get_clssifier_type_and_param_grid(classifier_name)

        if config.crossValidationType == "k-fold-cv":
            ####################################################################
            ############### classification with cross validation start ######################
            ####################################################################
            cv_clfs = [0] * Constant.NUM_FOLD_CV
            inner_roc_auc_scores = np.zeros(Constant.NUM_FOLD_CV)

            for i in range(0, Constant.NUM_FOLD_CV):
                X_train = X_trains[i]
                y_train = y_trains[i]
                # cv_clf, inner_roc_auc_score = my_CV.cross_validate_k_fold(classifier_name, clssifier_type, X_train, y_train, parameters)
                cv_clf = my_CV.cross_validate_hold_out_by_roc_auc(clssifier_type, X_train, y_train, parameters)
                # print("clf.cv_results_: ",clf.cv_results_)
                print("clf.best_params: ", cv_clf.best_params_)
                print("clf.best_score_ (ROC-AUC score): ", cv_clf.best_score_)
                print("clf.best_estimator_: ", cv_clf.best_estimator_)
                print("clf.best_index_: ", cv_clf.best_index_)

                # cv_clfs[i] = cv_clf
                # https://www.pynote.info/entry/sklearn-grid-search-cv
                cv_clfs[i] = cv_clf.best_estimator_
                inner_roc_auc_scores[i] = cv_clf.best_score_

            # choose the clf giving the largest auc roc
            best_clf = cv_clfs[inner_roc_auc_scores.argmax()]
            ###########################################################################
            ############### classification with cross validation end ######################
            ###########################################################################

            ###########################################################################
            ############### test the model start ####################
            ###########################################################################
            print("------------ start testing the model ------------")
            accuracies = np.zeros(Constant.NUM_FOLD_CV)
            f_measures = np.zeros(Constant.NUM_FOLD_CV)
            roc_auc_scores = np.zeros(Constant.NUM_FOLD_CV)

            for i in range(0, Constant.NUM_FOLD_CV):

                print("i = ", i)

                accuracy, f_measure, my_roc_auc_score = Test.test_model_k_fold(classifier_name, best_clf, X_tests[i], y_tests[i])

                accuracies[i] = accuracy
                f_measures[i] = f_measure
                roc_auc_scores[i] = my_roc_auc_score

            average_accuracy = np.mean(accuracies)
            average_f_measure = np.mean(f_measures)
            average_roc_auc_score = np.mean(roc_auc_scores)
            print("average_accuracy: ",average_accuracy )
            print("average_f_measure: ",average_f_measure )
            print("average_roc_auc_score: ",average_roc_auc_score )

            # append the test result for export
            file_name = os.path.basename(filePath)
            config.test_results.append([file_name ,classifier_name ,average_accuracy ,average_f_measure, average_roc_auc_score])
            print("------------ end testing the model ------------")
            ###########################################################################
            ############### test the model end ######################
            ###########################################################################

        elif config.crossValidationType == "hold-out":
            ####################################################################
            ############### classification with cross validation start ######################
            ####################################################################
            clf = my_CV.cross_validate_hold_out(clssifier_type, X_train, y_train, parameters)
            # print("clf.cv_results_: ",clf.cv_results_)
            print("clf.best_params: ",clf.best_params_ )
            print("clf.best_score_: ",clf.best_score_ )
            print("clf.best_estimator_: ",clf.best_estimator_)
            print("clf.best_index_: ",clf.best_index_)
            ###########################################################################
            ############### classification with cross validation end ######################
            ###########################################################################

            ###########################################################################
            ############### test the model start ####################
            ###########################################################################
            print("------------ start testing the model ------------")
            accuracy, f_measure, my_roc_auc_score = Test.test_model_hold_out(classifier_name, clf, X_test, y_test)

            # append the test result for export
            file_name = os.path.basename(filePath)
            config.test_results.append([file_name ,classifier_name ,accuracy ,f_measure ,my_roc_auc_score])
            print("------------ end testing the model ------------")
            ###########################################################################
            ############### test the model end ######################
            ###########################################################################

############### export the test result ###############
header = ["file_name","classifier","accuracy","f-measure","ROC-AUC"]
# https://note.nkmk.me/python-pathlib-name-suffix-parent/
Util.exportCSVFile(config.test_results, header, fileName=dataset_directory.parents[1].name + "_test_result")

################################################################
# Reference (not used)
# https://stackoverflow.com/questions/22165641/install-libsvm-for-python-in-windows
# http://tkoyama1988.hatenablog.com/entry/2013/12/09/125143
# author of this library https://www.csie.ntu.edu.tw/~cjlin/libsvm/#download

# sys.path.append('C:\\Users\\kensa\\Downloads\\libsvm-3.12\\python\\')
# from svm import *
# from svmutil import *
#
# data1 = [[1,-2,3],[1,-4,5,1234],[342,342,435345,43534,4352,-4]] #データ
# label1 = [-1,-1,-1] #データの正負（±１のラベル）
# prob = svm_problem(label1 , data1) # データとラベルを合成
# param = svm_parameter('-s 0 -t 0') # 学習方法の設定
# m = svm_train(prob, param) #学習
# result = svm_predict([-1,-1],[[34,453.5],[45356,-10]] , m) #未知データの適用
################################################################

################################################################
# Reference (not used)
# cross_validation package
# https://qiita.com/yhyhyhjp/items/d4b796f7658b7e5be3b6
################################################################
