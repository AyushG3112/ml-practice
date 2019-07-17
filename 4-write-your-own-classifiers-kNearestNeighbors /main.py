import random
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris # https://en.wikipedia.org/wiki/Iris_flower_data_set
from sklearn import tree

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, features, labels):
        self.X_train = features
        self.y_train = labels
        return self
    
    def predict(self, input):
        predictions = []
        for row in input:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_idx = 0
        for i, value in enumerate(self.X_train):
            dist = euc(row, value)
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        return self.y_train[best_idx]

iris=load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

# clf = tree.DecisionTreeClassifier()
clf = ScrappyKNN()

clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print accuracy_score(y_test, predictions)