from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris # https://en.wikipedia.org/wiki/Iris_flower_data_set
from sklearn import tree

iris=load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# clf = tree.DecisionTreeClassifier()
clf = KNeighborsClassifier()

clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print accuracy_score(y_test, predictions)