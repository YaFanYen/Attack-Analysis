# -*- coding: utf-8 -*-

import os
import numpy as np
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

with open("Logs/Train_10000/C_C.json",'r') as f:
    CC = f.read().splitlines()
CC = np.array(CC)
CC_length = len(CC)

with open("Logs/Train_10000/DDoS.json",'r') as f:
    DDoS = f.read().splitlines()
DDoS = np.array(DDoS)
DDoS_length = len(DDoS)

with open("Logs/Train_10000/IP_scan.json",'r') as f:
    IP_scan = f.read().splitlines()
IP_scan = np.array(IP_scan)
IP_scan_length = len(IP_scan)

with open("Logs/Train_10000/port_scan.json",'r') as f:
    port_scan = f.read().splitlines()
port_scan = np.array(port_scan)
port_scan_length = len(port_scan)

with open("Logs/Train_10000/RDP_bruteforce.json",'r') as f:
    RDP_bruteforce = f.read().splitlines()
RDP_bruteforce = np.array(RDP_bruteforce)
RDP_bruteforce_length = len(RDP_bruteforce)

x = []
x = np.append(CC, DDoS)
x = np.append(x, IP_scan)
x = np.append(x, port_scan)
x = np.append(x, RDP_bruteforce)
y1 = 1 * np.ones((CC_length,1))
y2 = 2 * np.ones((DDoS_length,1))
y3 = 3 * np.ones((IP_scan_length,1))
y4 = 4 * np.ones((port_scan_length,1))
y5 = 5 * np.ones((RDP_bruteforce_length,1))
y = np.append(y1, y2)
y = np.append(y, y3)
y = np.append(y, y4)
y = np.append(y, y5)

def train(algorithm, X_train, y_train):
    model = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', algorithm)])
    model.fit(X_train, y_train)
    return model

def result(actual, predictions):
    actual = np.array(actual)
    
    print(confusion_matrix(actual, predictions))
    print(classification_report(actual, predictions))
    print("Accuracy: " + str(round(accuracy_score(actual, predictions),2)))

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = train(ensemble.RandomForestClassifier(), X_train, y_train)

test_path = '309511051'
test_data = os.listdir(test_path)
test_len = len(test_data)

for i in range(test_len):
    category_path = test_path + '/' + test_data[i]
    with open(category_path,'r') as f:
        data = f.read().splitlines()
    category = test_data[i]
    a = np.array(data)
    predictions = model.predict(a)
    predictions = predictions.astype(int)
    predictions = np.argmax(np.bincount(predictions))
    if predictions == 1:
        print(category + ' : C&C')
    elif predictions == 2:
        print(category + ' : DDoS')
    elif predictions == 3:
        print(category + ' : IP_scan')
    elif predictions == 4:
        print(category + ' : port_scan')
    else:
        print(category + ' : RDP_bruteforce')

'''
predictions = model.predict(x_test)
print(predictions)
result(y_test, predictions)
'''
