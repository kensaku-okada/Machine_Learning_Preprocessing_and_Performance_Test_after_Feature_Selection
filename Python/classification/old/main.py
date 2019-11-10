#!/usr/bin/python3
#coding:utf-8

import os, sys, datetime, glob, pathlib
import numpy as np
# from scipy.io import arff as scipy_arff # for dense arff dataset
import pandas as pd
import Util, Constant
# import matplotlib.pyplot as plt
############### package for preprocessing ######################
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
############### package for preprocessing ######################
############### package for classification with cross validation ######################
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import tree
############### package for classification with cross validation ######################
############### package for testing ######################
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
############### package for testing ######################

# "outputDatasetType" = "densed" or "sparsed"
datasetType = {"importDatasetFormat": "arff", "outputDatasetType": "densed"}

########### specify the dataset start ##############
# if you want to process multiple files, active below
# ifGetMultipleResults = True
# relativePath = "datasets\\dorothea\\*bornfs.arff"

# if you want to process only one file, active below
ifGetMultipleResults = False
# relativePath = "datasets\\mushroom\\mRMR\\mushroom_mrmr.arff"
relativePath = "datasets\\arcene\\mRMR\\arcene-10-disc_mrmr.arff"
# relativePath = "datasets\\dorothea\\dorothea.sparse.arff.0.5.bornfs.arff"
########### specify the dataset end ##############

filePath = Util.getFilePath(relativePath)
print("filePath: ",filePath)
# https://note.nkmk.me/python-pathlib-name-suffix-parent/
dataset_directory = pathlib.Path(filePath)
print("dataset_directory: ",dataset_directory)
dataset_name = pathlib.Path(filePath).parents[1].name
print("dataset_name: ",dataset_name)

if ifGetMultipleResults:
    # print("glob.glob(filePath): ",glob.glob(filePath))
    file_paths = glob.glob(filePath)
else:
    # adjust the file format with the case ifGetMultipleResults = True
    file_paths = [filePath]

# compared classifiers
# classifiers = [Constant.SVC, Constant.NAIVE_BAYES, Constant.C4_5]
classifiers = [Constant.SVC]

test_results = []

for filePath in file_paths:

    print("filePath: ",filePath)

    # import data
    dataset = Util.importArffData(filePath, datasetType)
    print("type(dataset): ",type(dataset))
    print("dataset.shape: ",dataset.shape)
    print("dataset.head(5):{}".format(dataset.head(5)))

    # binarize the dataset
    # https://note.nkmk.me/python-pandas-get-dummies/
    dataset_binary = pd.get_dummies(dataset)
    print("type(dataset_binary): ",type(dataset_binary))
    print("dataset_binary.shape: ",dataset_binary.shape)
    print("dataset_binary.head(5):{}".format(dataset_binary.head(5)))

    ####################################################################
    ############### preprocessing start ###############
    ####################################################################

    X = dataset_binary.iloc[:, 0:len(dataset_binary.columns) - 1].values
    y = dataset_binary.iloc[:, len(dataset_binary.columns) - 1].values
    print("type(X): ", type(X))
    print("X.shape: ", X.shape)
    print("type(y): ", type(y))
    print("y.shape: ", y.shape)
    # print("y: ", y)
    # print("y[0][0]: ", y[0])
    # print("type(y[0][0]): ", type(y[0]))

    # # when processing mushroom_mrmr.arff, you want to use below
    # if dataset_name == "mushroom":
    #
    #     # https://it-ojisan.tokyo/numpy-where-all-any/
    #     # replace the label (output) in to 0 and 1
    #     y = np.where(y == "p", 1, 0)
    #
    #     # binarize the data to resolve the following error
    #     # https://towardsdatascience.com/building-a-perfect-mushroom-classifier-ceb9d99ae87e
    #     X_binary = pd.get_dummies(X)
    #
    #     # import itertools
    #     # # TypeError: unorderable types: NoneType() < str()
    #     # X_binary = np.empty((X.shape[0], X.shape[1]))
    #     # # source: https://www.datatechnotes.com/2019/05/one-hot-encoding-example-in-python.html
    #     # for i in range (0, X.shape[0]):
    #     #     x_numerical = LabelEncoder().fit_transform(X[i])
    #     #     print("x_numerical: ",x_numerical)
    #     #     onehot_encoder = OneHotEncoder(sparse=False)
    #     #     x_numerical_reshaped = x_numerical.reshape(len(x_numerical), 1)
    #     #     x_binary = onehot_encoder.fit_transform(x_numerical_reshaped)
    #     #     print("x_binary: ",x_binary)
    #     #     # change the 2 dim array into 1 array by concatenating the elements
    #     #     # https://note.nkmk.me/python-list-flatten/
    #     #     x_binary = list(itertools.chain.from_iterable(x_binary))
    #     #     X_binary[i] = x_binary
    #
    # else:
    #     # just change the data type of y from string to int
    #     y = y.astype(np.int32)
    #
    #     # binarize the data to resolve the following error
    #     # ValueError: You appear to be using a legacy multi-label data representation. Sequence of sequences are no longer supported; use a binary array or sparse matrix instead.
    #     # source: https://stackoverflow.com/questions/34213199/scikit-learn-multilabel-classification-valueerror-you-appear-to-be-using-a-leg
    #     # https://github.com/phanein/deepwalk/issues/32
    #     X_binary = MultiLabelBinarizer().fit_transform(X)
    #
    # print("X_binary: ", X_binary)
    # print("X_binary.shape: ", X_binary.shape)

    # print("y.shape: ", y.shape)
    # print("y]: ", y)
    # print("y[0]: ", y[0])
    # print("type(y[0]): ", type(y[0]))

    # divide data into training data and test data by 1:4 = test : train
    # if you want change the random pattern each time, remove random_state=0
    # random_state=: データを分割する際の乱数のシード値 (https://docs.pyq.jp/python/machine_learning/tips/train_test_split.html)
    # https://qiita.com/tomov3/items/039d4271ed30490edf7b
    # X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("X_train.shape: ", X_train.shape)
    # print("X_train: ", X_train)
    print("X_test.shape: ", X_test.shape)
    print("y_train.shape: ", y_train.shape)
    # print("y_train: ", y_train)
    print("y_test.shape: ", y_test.shape)

    # convert from Byte code to String, and from String to int, which is necessayr only when the date is byte type.
    # https://qiita.com/masakielastic/items/2a04aee632c62536f82c
    if type(y_train) is bytes:
        y_train = np.array([int(y.decode('utf-8')) for y in y_train])
        # print("y_train: ", y_train)
        # print("y_train.shape: ", y_train.shape)
        # print("type(y_train): ", type(y_train))
        # print("y_train[0]: ", y_train[0])
        # print("type(y_train[0]): ", type(y_train[0]))
    ####################################################################
    ############### preprocessing end ###############
    ####################################################################

    for classifier in classifiers:

        ####################################################################
        ############### classification with cross validation start ######################
        ####################################################################
        # syntax: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
        # parameters: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
        # C and γ is a practical method to identify good parameters (for example, C = 2^−5 , 2^−3 , . . . , 2^15 , γ = 2^−15 , 2^−13 , . . . , 2^3 ).
        C_params = [2**i for i in range(-5, 16 ,2) ]
        gamma_params = [2**i for i in range(-15, 4 ,2) ]
        print("C_params: ",C_params)
        print("gamma_params: ",gamma_params)
        parameters = {'kernel':['rbf'], 'C': C_params, 'gamma': gamma_params}
        # print("parameters: ",parameters)

        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        # if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.
        # svc = svm.SVC(gamma="scale")
        if classifier == Constant.SVC:
            svc = svm.SVC()
        elif classifier == Constant.NAIVE_BAYES:
            pass
        elif classifier == Constant.C4_5:
            pass
            # todo implement it referring to: https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart
            # https://www.quora.com/What-is-the-best-way-to-implement-C4-5-or-C5-0-algorithm-using-Python
        else:
            print("undefied classifier specified. stop the program")
            sys.exit(0)

        # GridSearchCV
        # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        # https://qiita.com/yhyhyhjp/items/c81f7cea72a44a7bfd3a
        # http://starpentagon.net/analytics/scikit_learn_grid_search_cv/
        clf = GridSearchCV(svc, parameters, cv=10)

        print ("start clf.fit at :{}".format(datetime.datetime.now()))
        clf.fit(X_train, y_train)
        print ("end clf.fit at :{}".format(datetime.datetime.now()))

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

        ################### calc F measure
        # get the beset parameters
        best_gamma = clf.best_params_['gamma']
        best_kernel = clf.best_params_['kernel']
        best_C = clf.best_params_['C']

        # https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155
        clf_test = svm.SVC(gamma = best_gamma, kernel=best_kernel, C=best_C).fit(X_test, y_test)
        y_pred = clf.predict(X_test)
        confusionMatrixResult = confusion_matrix(y_test, y_pred)
        # print("y_test: ",y_test)
        print("type(y_test): ",type(y_test))
        print("type(y_test[0]): ",type(y_test[0]))
        # print("y_pred: ",y_pred)
        print("type(y_pred): ",type(y_pred))
        print("type(y_pred[0]): ",type(y_pred[0]))
        print("confusionMatrixResult: ", confusionMatrixResult)
        print("Accuracy (<> F measure = f1score) is : ", clf.score(X_test, y_test))

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
        # https://www.haya-programming.com/entry/2019/03/18/052035
        # assuming 1 = positive, 0 = negative
        # print ("classification_report(y_test, y_pred, output_dict=False): ",classification_report(y_test, y_pred, target_names=["1", "0"], output_dict=False))
        print ("classification_report(y_test, y_pred, output_dict=False): ",classification_report(y_test, y_pred, output_dict=False))
        # my_classification_report = classification_report(y_test, y_pred, target_names=["1", "0"], output_dict=True)
        my_classification_report = classification_report(y_test, y_pred, output_dict=True)
        f_measure = my_classification_report['1']['f1-score']
        print ("F measure at 0 is : ",my_classification_report['0']['f1-score'])
        print ("F measure at 1 is : ",my_classification_report['1']['f1-score'])

        # this works too
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
        # f_measure = f1_score(y_test, y_pred)
        # print ("F measure is : ",f_measure)

        #########################################################
        ################### calc ROC-AUC
        # https://note.nkmk.me/python-sklearn-roc-curve-auc-score/
        fpr_all, tpr_all, thresholds_all = roc_curve(y_test, y_pred, drop_intermediate=False)
        my_roc_auc_score = roc_auc_score(y_test, y_pred)
        print ("ROC-AUC score is : ", my_roc_auc_score)

        # append the test result for export
        file_name = os.path.basename(filePath)
        accuracy = clf.score(X_test, y_test)
        test_results.append([file_name ,classifier ,accuracy ,f_measure ,my_roc_auc_score])
        print("------------ end testing the model ------------")
        ###########################################################################
        ############### test the model end ######################
        ###########################################################################

############### export the test result ###############
header = ["file_name","classifier","accuracy","f-measure","ROC-AUC"]
# https://note.nkmk.me/python-pathlib-name-suffix-parent/
Util.exportCSVFile(test_results, header, fileName=dataset_directory.parents[1].name + "_test_result")


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
