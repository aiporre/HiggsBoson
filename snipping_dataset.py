import pandas as pd
from sklearn import svm, preprocessing, cross_validation
import numpy as np

higgsboson = pd.read_csv('data/training.csv')
print higgsboson
# quinoa_bussiness.set_index('Date', inplace=True)
# X = np.array(quinoa_bussiness.iloc[:,:-3])
# X = preprocessing.scale(X)
# y = np.array(quinoa_bussiness.iloc[:,-1])
# X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
# print 'Classification score: ' + str(clf.score(X_test, y_test))





