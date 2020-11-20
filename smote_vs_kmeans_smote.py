#from scipy.sparse import linalg
import smote_variants as sv
import sklearn.datasets as datasets

"""

dataset= datasets.load_breast_cancer()

oversampler= sv.KernelADASYN()

X_samp, y_samp= oversampler.sample(dataset['data'], dataset['target'])
"""

import csv
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score, precision_score, f1_score
from statistics import mean

scoring = ['precision_macro', 'recall_macro', 'f1_macro']


def classification_and_report_generation(X, Y):
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

    #print(X_test)
    #print(y_test)
    
    svm = SVC(kernel='rbf', random_state=1, gamma=0.1, C=0.02)
    scores = cross_validate(svm, X, Y, scoring=scoring, cv=5)
    print('average test precision: ' + str(mean(scores['test_precision_macro'])))
    print('average test recall: ' +str(mean(scores['test_recall_macro'])))
    print('average test F1 score: ' + str(mean(scores['test_f1_macro'])))
    
    #svm.fit(X_train, y_train)
    
    #y_pred = svm.predict(X_test)
    
    #print(classification_report(y_test, y_pred))

#dermatology dataset, here the features are integer values
print("dermatology dataset")

X = []
Y = []

with open("dermatology-6.csv", 'r') as file:
    csv_file = csv.reader(file)
    next(csv_file, None)
    for row in csv_file:
        label = row[-1]   # label, either 'negative' or 'positive'
        #print(label)
        if label == 'negative':
            Y.append(0)
        else:
            Y.append(1)
        
        row = [int(i) for i in row[:-1]]   # convert string to int except the last column
        X.append(row)
        
#print(X)
#print(Y)
scaler = StandardScaler()
# fit and transform the data
X = scaler.fit_transform(X) # standardizing the data

X = np.array(X)
Y = np.array(Y)
# without numpy array the oversampler shows error.

# classification_and_report_generation(X, Y)

#after SMOTE
print("Now SMOTE will be applied")
oversampler= sv.SMOTE()
X_samp, y_samp= oversampler.sample(X, Y)

classification_and_report_generation(X_samp, y_samp)

#after kmeans_SMOTE
print("Now kmeans_SMOTE will be applied")
oversampler= sv.kmeans_SMOTE()
X_samp, y_samp= oversampler.sample(X, Y)

classification_and_report_generation(X_samp, y_samp)


# now ecoli dataset, here the features are floating point values
print("ecoli dataset")

X = []
Y = []

with open("ecoli.csv", 'r') as file:
    csv_file = csv.reader(file)
    next(csv_file, None)
    for row in csv_file:
        label = row[-1]   # label, either 0 or 1
        #print(label)
        Y.append(int(label))
        
        row = [float(i) for i in row[:-1]]   # convert string to int except the last column
        X.append(row)
        
#print(X)
#print(Y)

scaler = StandardScaler()
# fit and transform the data
X = scaler.fit_transform(X) # standardizing the data

X = np.array(X)
Y = np.array(Y)
# without numpy array the oversampler shows error.

# classification_and_report_generation(X, Y)

#after SMOTE
print("Now SMOTE will be applied")
oversampler= sv.SMOTE()
X_samp, y_samp= oversampler.sample(X, Y)

classification_and_report_generation(X_samp, y_samp)

#after kmeans_SMOTE
print("Now kmeans_SMOTE will be applied")
oversampler= sv.kmeans_SMOTE()
X_samp, y_samp= oversampler.sample(X, Y)

classification_and_report_generation(X_samp, y_samp)


