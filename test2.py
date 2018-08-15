import _pickle as c
import os
from sklearn import *
from collections import Counter


def load(clf_file):
    with open(clf_file, 'rb') as fp:
        clf = c.load(fp)
    return clf


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


clf = load("text-classifier.mdl")
d = make_dict()

f = open('TestReview.txt', encoding="utf8")
lines = [line.rstrip('\n') for line in open('TestReview.txt', encoding="utf8")]
for line in lines:
    features = []
    fulline = line
    line = line.split()
    for word in d:
        features.append(line.count(word[0]))
    res = clf.predict([features])
    print(fulline)
    print (["Negative", "Positive"][res[0]])
    print("\n")


while True:
    features = []
    inp = input(">").split()
    if inp[0] == "exit":
        break
    for word in d:
        features.append(inp.count(word[0]))
    res = clf.predict([features])
    print (["Negative", "Positive"][res[0]])
