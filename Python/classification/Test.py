#!/usr/bin/python3
#coding:utf-8

import os, sys, datetime, glob, pathlib
import numpy as np
# from scipy.io import arff as scipy_arff # for dense arff dataset
import pandas as pd
import Util, Constant
############### package for testing ######################
from sklearn import svm
from sklearn.metrics import f1_score, classification_report, roc_auc_score, roc_curve, confusion_matrix
from sklearn.naive_bayes import GaussianNB
############### package for testing ######################

def test_model_k_fold(classifier_name, clf, X_test, y_test):

    y_pred = get_predicted_y(classifier_name, clf, X_test)
    # print("y_test: ",y_test)
    print("type(y_test): ",type(y_test))
    print("type(y_test[0]): ",type(y_test[0]))
    # print("y_pred: ",y_pred)
    print("type(y_pred): ",type(y_pred))
    print("type(y_pred[0]): ",type(y_pred[0]))

    confusionMatrixResult = confusion_matrix(y_test, y_pred)
    print("confusionMatrixResult: ", confusionMatrixResult)

    accuracy = clf.score(X_test, y_test)
    print("Accuracy (<> F measure = f1score) is : ", accuracy)

    my_classification_report = classification_report(y_test, y_pred, output_dict=True)
    print("my_classification_report: ",my_classification_report)

    f_measure = my_classification_report['1']['f1-score']
    print ("classification_report(y_test, y_pred, output_dict=False): ",classification_report(y_test, y_pred, output_dict=False))
    print ("F measure at 0 is : ",my_classification_report['0']['f1-score'])
    print ("F measure at 1 is : ",f_measure)

    # my_roc_auc_score = roc_auc_score(y_test, y_pred)
    # print ("ROC-AUC score is : ", my_roc_auc_score)

    return confusionMatrixResult, accuracy, f_measure


def test_model_hold_out(classifier_name, clf, X_test, y_test):

    y_pred = get_predicted_y(classifier_name, clf, X_test)

    # print("y_test: ",y_test)
    print("type(y_test): ",type(y_test))
    print("type(y_test[0]): ",type(y_test[0]))
    # print("y_pred: ",y_pred)
    print("type(y_pred): ",type(y_pred))
    print("type(y_pred[0]): ",type(y_pred[0]))

    confusionMatrixResult = confusion_matrix(y_test, y_pred)
    print("confusionMatrixResult: ", confusionMatrixResult)

    #########################################################
    ################### get accuracy ###################
    #########################################################
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    # is this correct?
    accuracy = clf.score(X_test, y_test)
    # is this correct?
    # accuracy = clf_test.score(X_test, y_test)
    print("Accuracy (<> F measure = f1score) is : ", accuracy)

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
    # https://www.haya-programming.com/entry/2019/03/18/052035
    # assuming 1 = positive, 0 = negative
    my_classification_report = classification_report(y_test, y_pred, output_dict=True)
    f_measure = my_classification_report['1']['f1-score']
    # print ("classification_report(y_test, y_pred, output_dict=False): ",classification_report(y_test, y_pred, target_names=["1", "0"], output_dict=False))
    print ("classification_report(y_test, y_pred, output_dict=False): ",classification_report(y_test, y_pred, output_dict=False))
    print ("F measure at 0 is : ",my_classification_report['0']['f1-score'])
    print ("F measure at 1 is : ",my_classification_report['1']['f1-score'])
    # this works too
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    # f_measure = f1_score(y_test, y_pred)
    # print ("F measure is : ",f_measure)

    #########################################################
    ################### get ROC-AUC ###################
    #########################################################
    # https://note.nkmk.me/python-sklearn-roc-curve-auc-score/
    fpr_all, tpr_all, thresholds_all = roc_curve(y_test, y_pred, drop_intermediate=False)
    # this parameter substitution is wrong !!!!!!!!!!!!!!
    my_roc_auc_score = roc_auc_score(y_test, y_pred)
    print ("ROC-AUC score is : ", my_roc_auc_score)

    return accuracy, f_measure, my_roc_auc_score


def get_predicted_y(classifier_name, clf, X_test):

    if classifier_name == Constant.SVC:
        # # get the beset parameters
        # best_gamma = clf.best_params_['gamma']
        # best_kernel = clf.best_params_['kernel']
        # best_C = clf.best_params_['C']

        # https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155
        # todo is this correct?
        # clf_test = svm.SVC(gamma=best_gamma, kernel=best_kernel, C=best_C).fit(X_test, y_test)
        y_pred = clf.predict(X_test)
        # is this correct?
        # clf_test = svm.SVC(gamma=best_gamma, kernel=best_kernel, C=best_C).fit(X_train, y_train)
        # y_pred = clf_test.predict(X_test)

    elif classifier_name == Constant.NAIVE_BAYES:

        # clf_test = GaussianNB().fit(X_test, y_test)
        y_pred = clf.predict(X_test)

    elif classifier_name == Constant.C4_5:
        clf_test = None
        y_pred = None
        # todo implement it referring to: https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart
        # https://www.quora.com/What-is-the-best-way-to-implement-C4-5-or-C5-0-algorithm-using-Python

    else:
        print("undefied classifier_name specified. stop the program")
        sys.exit(0)

    return y_pred

def get_overall_accuracy(confusionMatrixes):
    # https://note.nkmk.me/python-sklearn-confusion-matrix-score/
    TN = sum([confusionMatrix[0][0] for confusionMatrix in confusionMatrixes])
    FN = sum([confusionMatrix[1][0] for confusionMatrix in confusionMatrixes])
    FP = sum([confusionMatrix[0][1] for confusionMatrix in confusionMatrixes])
    TP = sum([confusionMatrix[1][1] for confusionMatrix in confusionMatrixes])

    # source: https://www.python-course.eu/confusion_matrix.php
    overall_accuracy = (TN+TP) / (TN+FN+FP+TP)

    return overall_accuracy


def get_overall_f_measure(confusionMatrixes):
    # https://note.nkmk.me/python-sklearn-confusion-matrix-score/
    TN = sum([confusionMatrix[0][0] for confusionMatrix in confusionMatrixes])
    FN = sum([confusionMatrix[1][0] for confusionMatrix in confusionMatrixes])
    FP = sum([confusionMatrix[0][1] for confusionMatrix in confusionMatrixes])
    TP = sum([confusionMatrix[1][1] for confusionMatrix in confusionMatrixes])

    overall_f_measure = 2.0 * TP / (2.0 * TP + FN + FP)

    # # source: https://www.python-course.eu/confusion_matrix.php
    # precision = TP / (TP+FP)
    # recall = TP / (TP+FN)
    #
    # # https://medium.com/hugo-ferreiras-blog/confusion-matrix-and-other-metrics-in-machine-learning-894688cb1c0a
    # overall_f_measure = 2.0*precision*recall / (precision+recall)

    return overall_f_measure
