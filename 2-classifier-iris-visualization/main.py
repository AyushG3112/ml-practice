import numpy as np
from sklearn.datasets import load_iris # https://en.wikipedia.org/wiki/Iris_flower_data_set
from sklearn import tree
iris=load_iris()
# print iris.feature_names
# print iris.data[0]
# print iris.target_names
# print iris.target[0]

test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0) # What is axis?

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)

predicted = clf.predict(test_data)

if predicted.all() == test_target.all():
    print "Predictions were right"
else:
    print "Predictions were wrong"


# visualization
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("iris") 
