# these are the python lines for the Week 8 lab, the first lab on classification

import nltk

# define a feature extraction function for each name
def gender_features(word):
    return{'last_letter': word[-1]}

print(gender_features('Shrek'))

# resource for male and female first names
from nltk.corpus import names
print(names.words('male.txt')[:20])
print(names.words('female.txt')[:20])

# make list of male and female names paired with gender
namesgender = ([(name, 'male') for name in names.words('male.txt')] +
          [(name, 'female') for name in names.words('female.txt')])
print(len(namesgender))
print(namesgender[:20])
print(namesgender[7924:])

# put the list into random order
import random
random.shuffle(namesgender)
print(namesgender[:20])

# featuresets represent each name as features and a label
featuresets = [(gender_features(n), g) for (n, g) in namesgender]
print(featuresets[:20])

# create training and test sets, run a classifier and show the accuracy
train_set, test_set = featuresets[500:], featuresets[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# classify new instances
print(classifier.classify(gender_features('Neo')))
print(classifier.classify(gender_features('Trinity')))

# classify accuracy function runs the classifier on the test set and reports
#   comparisons between predicted labels and actual/gold labels
print(nltk.classify.accuracy(classifier, test_set))


# this function available for naive bayes classifiers
print(classifier.show_most_informative_features(20))

# creating lots of features
#   there are probably too many features but we are demonstrating different
#     types of features
def gender_features2(name):
    features = {}
    features["firstletter"] = name[0].lower()
    features["lastletter"] = name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features["count({})".format(letter)] = name.lower().count(letter)
        features["has({})".format(letter)] = (letter in name.lower())
    return features

features = gender_features2('Shrek')
print(len(features))
print(features)

# create feature sets using this function
featuresets2 = [(gender_features2(n), g) for (n, g) in namesgender]

for (n, g) in namesgender[:3]:
    print(n, gender_features2(n), '\n')

# create new training and test sets, classify and look at accuracy
train_set, test_set = featuresets2[500:], featuresets2[:500]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier, test_set))


## Experiment
# go back and separate the names into training and test
train_names = namesgender[500:]
test_names = namesgender[:500]

# use our original features to train a classify and test on the development test set
train_set = [(gender_features(n), g) for (n, g) in train_names]
test_set = [(gender_features(n), g) for (n, g) in test_names]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# save the classifier accuracy for use in the exercise
print(nltk.classify.accuracy(classifier, test_set))

# define a function that will compare the classifier labels with the gold standard labels
def geterrors(test):
    errors = []
    for (name, tag) in test:
        guess = classifier.classify(gender_features(name))
        if guess != tag:
            errors.append( (tag, guess, name) )
    return errors

errors = geterrors(test_names)
print(len(errors))

# define a function to print the errors
def printerrors(errors):
    for (tag, guess, name) in sorted(errors):
        print('correct={:<8s} guess={:<8s} name={:<30s}'.format(tag, guess, name))

printerrors(errors)

# evaluation measures showing performance of classifier

from nltk.metrics import *

reflist = []
testlist = []
for (features, label) in test_set:
    reflist.append(label)
    testlist.append(classifier.classify(features))

print(reflist[:30])
print(testlist[:30])

# define and print confusion matrix

cm = ConfusionMatrix(reflist, testlist)
print(cm)

# define a set of item identifiers that are gold labels and a set of item identifiers that are predicted labels
# this uses index numbers for the labels

reffemale = set([i for i,label in enumerate(reflist) if label == 'female'])
refmale = set([i for i,label in enumerate(reflist) if label == 'male'])
testfemale = set([i for i,label in enumerate(testlist) if label == 'female'])
testmale = set([i for i,label in enumerate(testlist) if label == 'male'])

reffemale
testfemale
refmale
testmale

# compute precision, recall and F-measure for each label

def printmeasures(label, refset, testset):
    print(label, 'precision:', precision(refset, testset))
    print(label, 'recall:', recall(refset, testset)) 
    print(label, 'F-measure:', f_measure(refset, testset))

printmeasures('female', reffemale, testfemale)
printmeasures('male', refmale, testmale)


# another feature extraction function for the exercise
def gender_features3(word):
    return {'suffix1': word[-1],'suffix2': word[-2]}

print(gender_features3('Shrek'))