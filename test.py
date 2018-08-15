import os
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
import _pickle as c


def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print("saved")


def make_dict():
	#read negative reviews
    direc = "negativeReviews/"
    files = os.listdir(direc)
    reviews = [direc + review for review in files]
    words = []
    c = len(reviews)
    for review in reviews:
        f = open(review, encoding="utf8")
        blob = f.read()
        words += blob.split(" ")
        print(c)
        c -= 1

    #read positive reviews
    direc = "positiveReviews/"    
    files = os.listdir(direc)
    reviews = [direc + review for review in files]
    c = len(reviews)
    for review in reviews:
        f = open(review, encoding="utf8")
        blob = f.read()
        words += blob.split(" ")
        print(c)
        c -= 1

    for i in range(len(words)):
        if not words[i].isalpha():
            words[i] = ""

    dictionary = Counter(words)
    del dictionary[""]
    return dictionary.most_common(30000)


def make_dataset(dictionary):
    direc = "negativeReviews/"
    files = os.listdir(direc)
    reviews = [direc + review for review in files]
    feature_set = []
    labels = []
    c = len(reviews)

    for review in reviews:
        data = []
        f = open(review,  encoding="utf8")
        words = f.read().split(' ')
        for entry in dictionary:
            data.append(words.count(entry[0]))
        feature_set.append(data)
        labels.append(0)
        print(c)
        c = c - 1

    direc = "positiveReviews/"
    files = os.listdir(direc)
    reviews = [direc + review for review in files]
    c = len(reviews)

    for review in reviews:
        data1 = []
        f = open(review,  encoding="utf8")
        words = f.read().split(' ')
        for entry in dictionary:
            data1.append(words.count(entry[0]))
        feature_set.append(data1)
        labels.append(1)
        print(c)
        c = c - 1

    return feature_set, labels


d = make_dict()
features, labels = make_dataset(d)


print(len(features))

x_train, x_test, y_train, y_test = tts(features, labels, test_size=0.2)

clf = MultinomialNB()
clf.fit(x_train, y_train)

preds = clf.predict(x_test)
print(accuracy_score(y_test, preds))
save(clf, "text-classifier.mdl")