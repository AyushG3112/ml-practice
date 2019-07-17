from sklearn import tree

features = [[140, "smooth"], [150, "smooth"], [160, "bumpy"], [170, "bumpy"]]
labels = ["apple", "apple", "orange", "orange"]

predictioLabels = {
    0: "apple",
    1: "oranges",
    2: "new discovery",
}

def cleanup_single_feature(feature):
    if feature[1] == "smooth":
        feature[1] = 1
    elif feature[1] == "bumpy":
        feature[1] = 2
    else:
        feature[1] = 3
    return feature

def cleanup_features(features):
    for i, x in enumerate(features):
        features[i] = cleanup_single_feature(x)
    return features

def cleanup_labels(labels):
    for i, x in enumerate(labels):
        print i, x
        if x == "apple":
            labels[i] = 0
        elif x == "orange":
            labels[i] = 1
        else:
            labels[i] = 2
    return labels

def convert_predictions_to_readable(predictions):
    return [predictioLabels[x] for x in predictions]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(cleanup_features(features), cleanup_labels(labels))
print convert_predictions_to_readable(clf.predict(cleanup_features([[100, 'smooth']])))