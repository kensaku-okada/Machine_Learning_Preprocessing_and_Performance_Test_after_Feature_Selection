#!/usr/bin/python3
#coding:utf-8


import datetime
import numpy as np
############### package for classification with cross validation ######################
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
# from sklearn import tree
from sklearn.metrics import make_scorer, accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score,roc_auc_score
############### package for classification with cross validation ######################
import Util, Constant

class my_CrossValidation:

	parameters = None

	def __init__(self, parameters):
		self.parameters = parameters


def get_clssifier_type_and_param_grid(classifier_name):

    if classifier_name == Constant.SVC:
        # syntax: https://scikit-learn.org/stable/auto_examples/model_selection/plot_grid_search_digits.html
        # parameters: https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
        # C and γ is a practical method to identify good parameters (for example, C = 2^−5 , 2^−3 , . . . , 2^15 , γ = 2^−15 , 2^−13 , . . . , 2^3 ).
        C_params = [2 ** i for i in range(-5, 16, 2)]
        gamma_params = [2 ** i for i in range(-15, 4, 2)]
        print("C_params: ", C_params)
        print("gamma_params: ", gamma_params)
        parameters = {'kernel': ['rbf'], 'C': C_params, 'gamma': gamma_params}
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        # if gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma.
        # svc = svm.SVC(gamma="scale")

        classifier_type = svm.SVC()

    elif classifier_name == Constant.NAIVE_BAYES:
        var_smoothing = [1e-09]
        parameters = {'var_smoothing': var_smoothing}

        # https://qiita.com/ynakayama/items/ca3f5e9d762bbd50ad1f
        # https://www.datacamp.com/community/tutorials/naive-bayes-scikit-learn
        classifier_type = GaussianNB()

    elif classifier_name == Constant.C4_5:
        classifier_type = None
        parameters = None
        # todo implement it referring to: https://scikit-learn.org/stable/modules/tree.html#tree-algorithms-id3-c4-5-c5-0-and-cart
        # https://www.quora.com/What-is-the-best-way-to-implement-C4-5-or-C5-0-algorithm-using-Python
    else:
        print("undefied classifier_name specified. stop the program")
        sys.exit(0)

    return classifier_type, parameters

def cross_validate_k_fold(model_obj, X, y, cv_obj, fit_params):
    '''
    reference:
        https://www.kaggle.com/questions-and-answers/30560
        https://www.programcreek.com/python/example/91147/sklearn.model_selection.StratifiedKFold
        https://qiita.com/KROYO/items/66d613356e2bf4cf9fed
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html
        https://qiita.com/Mukomiz/items/f5c562ff8b538c1502d7

        https://qiita.com/kenmatsu4/items/0a862a42ceb178ba7155
        https://blog.amedama.jp/entry/sklearn-cv-custom-metric
    '''

    # ここから


    # https://stackoverflow.com/questions/46598301/how-to-compute-precision-recall-and-f1-score-of-an-imbalanced-dataset-for-k-fold
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
    scoring = {'accuracy': make_scorer(accuracy_score),
               'precision': make_scorer(precision_score),
               'recall': make_scorer(recall_score),
               'f1_score': make_scorer(f1_score),
               'roc_auc_score': make_scorer(roc_auc_score),
               'kappa': make_scorer(cohen_kappa_score),
               }

    print("start cross_validate at :{}".format(datetime.datetime.now()))
    scores_skf = cross_validate(estimator=model_obj, X=X, y=y,
                            cv=cv_obj,
                            n_jobs = -1,
                            fit_params = fit_params,
                            scoring=scoring)
    print("end cross_validate at :{}".format(datetime.datetime.now()))

    return scores_skf







