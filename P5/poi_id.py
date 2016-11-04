#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier
from tester import dump_classifier_and_data
from sklearn.pipeline import Pipeline
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','long_term_incentive','total_payments','total_stock_value',
                 'deferral_payments','deferred_income','director_fees',
                 'exercised_stock_options','expenses','from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','loan_advances','long_term_incentive','other',
                 'restricted_stock','restricted_stock_deferred','shared_receipt_with_poi','to_messages'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

features= ['poi','salary','bonus','long_term_incentive','total_payments','total_stock_value',
                 'deferral_payments','deferred_income','director_fees',
                 'exercised_stock_options','expenses','from_messages','from_poi_to_this_person',
                 'from_this_person_to_poi','loan_advances','long_term_incentive','other',
                 'restricted_stock','restricted_stock_deferred','shared_receipt_with_poi','to_messages'] # You will need to use more features


data = featureFormat(data_dict, features)
'''
for point in data:
    salary = point[1]
    bonus = point[2]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()

for i in data_dict:
    if data_dict[i]['salary'] != 'NaN' and data_dict[i]['salary'] > 25000000:
        print i
      '''
#the outlier is 'TOTAL' and this is not a person, so i don't need it.

#Deleting outlier 
data_dict.pop('TOTAL')

#see whether it works well

data = featureFormat(data_dict, features)

'''
for point in data:
    salary = point[1]
    bonus = point[2]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
 
#search for another outlier

for point in data:
    salary = point[1]
    total_payments = point[4]
    matplotlib.pyplot.scatter( salary, total_payments )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("total_payments")
matplotlib.pyplot.show()

for i in data_dict:
    if data_dict[i]['total_payments'] != 'NaN' and data_dict[i]['total_payments'] > 100000000:
        print i
#the outlier is LAY KENNETH L, so this outlier is important information.

for point in data:
    salary = point[1]
    loan_advances = point[14]
    matplotlib.pyplot.scatter( salary, loan_advances )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("loan_advances")
matplotlib.pyplot.show()

for i in data_dict:
    if data_dict[i]['loan_advances'] != 'NaN' and data_dict[i]['loan_advances'] > 7000000:
        print i
#Again the outlier is LAY KENNETH L
'''

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for i in my_dataset:
    if my_dataset[i]['from_poi_to_this_person'] != 'NaN' and\
    my_dataset[i]['from_this_person_to_poi'] != 'NaN' and\
    my_dataset[i]['to_messages'] != 'NaN' and\
    my_dataset[i]['from_messages'] != 'NaN':
        my_dataset[i]['percentage_from_poi_to_entire_email'] = (my_dataset[i]['from_poi_to_this_person']+my_dataset[i]['from_this_person_to_poi'])/float((my_dataset[i]['from_messages']+my_dataset[i]['to_messages']))*100
    else:
        my_dataset[i]['percentage_from_poi_to_entire_email'] = 0

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
'''
#GaussianNB
clf = GaussianNB()
clf.fit(features,labels)
pred = clf.predict(features)
precision = precision_score(labels,pred)
recall = recall_score(labels,pred)
f1 = f1_score(labels,pred)
print precision
print recall
print f1

#DecisionTreeClassifer
clf = DecisionTreeClassifier()
clf.fit(features,labels)
pred = clf.predict(features)
test_classifier(clf, data_dict, features_list)

#AdaBoost

clf = AdaBoostClassifier()
clf.fit(features,labels)
pred = clf.predict(features)
test_classifier(clf, data_dict, features_list)


'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#features_train, features_test, labels_train, labels_test = \
#    train_test_split(features, labels, test_size=0.3, random_state=42)
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA


#split data
from sklearn.cross_validation import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(labels, 10,test_size = 0.4, random_state = 90)

for sss_train, sss_test in sss:
    features_train = []
    features_test = []
    labels_train = []
    labels_test = []
    #creates train,test data
    for ii in sss_train:
        features_train.append(features[ii])
        labels_train.append(labels[ii])
    for jj in sss_test:
        features_test.append(features[jj])
        labels_test.append(labels[jj])


#Tuning classifier
from sklearn.feature_selection import SelectKBest


kbest = SelectKBest()
scaler = MinMaxScaler()
ada = AdaBoostClassifier()
nb = GaussianNB()
cv =StratifiedShuffleSplit(labels_train, 90,test_size = 0.3,random_state=50)
pca = PCA()

parameters = {'kbest__k':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'scaler__copy':[True, False]}
pipe = Pipeline([('scaler',scaler),('kbest',kbest),('ada',ada)])
grid_search = GridSearchCV(pipe,parameters, cv=cv, scoring = 'f1')
clf =grid_search.fit(features_train,labels_train)
clf = grid_search.best_estimator_
test_classifier(clf, data_dict, features_list)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
#test_classifier(clf, data_dict, features_list)
dump_classifier_and_data(clf, my_dataset, features_list)
