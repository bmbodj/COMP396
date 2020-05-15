"""## Evaluate_embeddings.py"""

#Evaluate embedding class 
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

"""Evaluates latent space quality via a linear regression downstream task."""
def linear_reg(X,Y,std):

  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
  params = { 'kernel':['laplacian'],'alpha':[0.001, 0.01,0.1,1,10,100]}
  #clf = KernelRidge(alpha=1.0, kernel='laplacian',)
  clf = GridSearchCV(KernelRidge(), params, cv=5)
  clf.fit(X_train, y_train)
  y_pred=clf.predict(X_test)
  mae = mean_absolute_error(y_test,y_pred)*std 
  print('Kernel Ridge Regression Mean Absolute Error : {}'.format(mae))
  return mae


def evaluate_embedding(embeddings, labels, std):
    x, y = np.array(embeddings), np.array(labels)
    print(x.shape, y.shape)

    linreg_accuracies = [linear_reg(x, y, std) for _ in range(1)]
    #print('LinReg', np.mean(linreg_accuracies))

    return np.mean(linreg_accuracies)
