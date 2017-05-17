import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1],[5,8],[7,9],[9,8]])
y = np.array([1, 1, 2, 2,3,3,3])
from sklearn.svm import SVC
clf = SVC(kernel='linear')
clf.fit(X, y) 
print(clf.predict([[6, 10]]))
print(clf.coef_)
print(clf.score(X,y))