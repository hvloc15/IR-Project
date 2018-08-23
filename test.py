import os
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import _pickle as c     

count_vector = CountVectorizer(binary=True, max_features=30000)
tfidf_transformer = TfidfTransformer()

def save(clf, name):
    with open(name, 'wb') as fp:
        c.dump(clf, fp)
    print("saved")

def read_files_in_directory(dir_name):
    files_names = os.listdir(dir_name)
    files_names = [dir_name + file_name for file_name in files_names]
    words = []
    for file_name in files_names:
        f = open(file_name, encoding="utf8")
        blob = f.read()
        words += blob.split(" ")
    return words
   
def get_features_targets():
    labels = []
	#read negative reviews
    negative_words = read_files_in_directory("negativeReviews/")
    labels = [0]*len(negative_words)
    #read positive reviews
    positive_words = read_files_in_directory("positiveReviews/")
    labels+=[1]*len(positive_words)
    
    words = negative_words + positive_words  

    return words, labels

def get_count_matrix(features):
    train_count = count_vector.fit_transform(features)
    return train_count
    
def get_tfidf_representation(count_matrix):
    x_train_tfidf = tfidf_transformer.fit_transform(count_matrix)
    return x_train_tfidf

def SaveDict(dictionary):
    pickle_out = open("dict.pickle","wb")
    c.dump(dictionary, pickle_out)
    pickle_out.close()

def LoadDict(dictionary):
    pickle_in = open("dict.pickle","rb")
    example_dict = c.load(pickle_in)
    pickle_in.close()
    


print("---------------------Start-------------------------")
print("Read files.....")
features, labels = get_features_targets()

count_vector = CountVectorizer(binary=True)
train_count = count_vector.fit_transform(features)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(train_count)

print("Fitting....")
clf = MultinomialNB().fit(X_train_tfidf, labels)
save(clf, "text-classifier.mdl")

print("Predict")

docs_new = []

f = open('TestReview.txt', encoding="utf8")
lines = [line.rstrip('\n') for line in open('TestReview.txt', encoding="utf8")]
for line in lines:
    fulline = line
    docs_new.append(line)

X_new_counts = count_vector.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

label = ['Negative Review', 'Positive Review']

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, label[category]))
