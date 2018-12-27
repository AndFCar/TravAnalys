#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Matplotlib visualization
import matplotlib.pyplot as plt


def fit_and_evaluate(model, name, X, y, X_test, y_test, X_valid, y_valid, cutoff):
           
    model.fit(X, y)
    
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)
    y_test_prob0 = y_test_prob[:,1]
    y_test_pred = (np.where(y_test_prob0>cutoff,1,0)) 

    y_valid_pred = model.predict(X_valid)
    y_valid_prob = model.predict_proba(X_valid)
    y_valid_prob0 = y_valid_prob[:,1]
    y_valid_pred = (np.where(y_valid_prob0>cutoff,1,0)) 
   
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)
    y_prob0 = y_prob[:,1]
    y_pred = (np.where(y_prob0>cutoff,1,0))
    
    ps = precision_score(y_test, y_test_pred)
    rs = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    AUC = roc_auc_score(y_test, y_test_prob0)
    
    print(name, 'Precicion:=%0.4f' % ps, ', Detects:=%0.4f' % rs, ', F1=%0.4f:' % f1, ', AUC=%0.4f:' % AUC)
    
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_prob0)
    
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Detects", linewidth=2)
    plt.xlabel("Threshold", fontsize=12)
    plt.legend(loc="upper left", fontsize=12)
    plt.ylim([0, 1])
    
    plt.show()
    
    df = pd.DataFrame(y_test_prob0)
    df.hist(bins=100)
    
    plt.show()
    
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob0)
    
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    
    plt.show()
    
    return y_prob0, y_test_prob0, y_valid_prob0





