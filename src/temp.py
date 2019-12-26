import numpy as np
import Util, Constant, sys
############### package for preprocessing ######################
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
############### package for preprocessing ######################



a = [[1,0,0],[2,1,2],[0,2,1]]
print("a:")
print(OneHotEncoder(categories='auto').fit_transform(a).toarray())

print("----------------------------------------")

a_str = [["1","0","0"],["2","1","2"],["0","2","1"]]
print("a_str:")
print(MultiLabelBinarizer().fit_transform(a_str))

print("----------------------------------------")

b = [[1,0,0],[2,1,2],[2,2,1]]
print("b:")
print(MultiLabelBinarizer().fit_transform(b))

print("----------------------------------------")

c = [[0,1,2],[3,4,5],[6,7,8]]
print("c:")
print(OneHotEncoder(categories='auto').fit_transform(c).toarray())

print("----------------------------------------")

d = [[0,1,2],[3,4,5],[6,7,8]]
print("d:")
print(MultiLabelBinarizer().fit_transform(d))


print("----------------------------------------")
c = [[0,1,2],[3,4,5],[6,7,7]]
print("c:")
print(OneHotEncoder(categories='auto').fit_transform(c).toarray())
d = [[0,1,2],[3,4,5],[6,7,7]]
print("d:")
print(MultiLabelBinarizer().fit_transform(d))

print("----------------------------------------")
