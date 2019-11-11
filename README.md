# Machine_Learning_Preprocessing_and_Performance_Test_after_Feature_Selection

This repository stores the data and python source codes evaluating a new feature selection algorithm.

The in-progress source code (from preprocessing to testing) is at: 
https://github.com/kensaku-okada/Machine_Learning_Preprocessing_and_Performance_Test_after_Feature_Selection/tree/master/Python/classification

## Pesudo code
k = 10
data = data[1] + ... + data[k] # k-foldによるデータの分解
for i in range(k):
    test = data[i]
    training = data - data[i]
    clf = new Clf
    params = optimalParams(clf, training)
    clf.setParams(params)
    clf.fit(training)
    prediction[i] = clf.predict(test)
computeAUC(prediction[1], ..., prediction[k])

def optimalParams(clf, training):
    training = tr[1] + ... + tr[k] # k-foldによるデータの分解
    for params in paramsSpace:
        clf.setParams(params)
    	for i in range(k):
    	    _test = tr[i]
	    _training = training - tr[i]
	    clf.fit(_training)
	    _prediction[params, i] = clf.predict(test)
    	vc_score[params] = computeAUC(_prediction[1], ..., _prediction[k])
    return argmax([vc_score[params] for params in paramsSpace])
