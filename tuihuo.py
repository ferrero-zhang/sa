# -*- coding: utf-8 -*-


# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import map, range, zip
import six
from pymongo import MongoClient
# 3rd party
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.cluster.vq as scv
import scipy.sparse as sp
import sklearn.cluster as skc
from sklearn.grid_search import ParameterGrid
from sklearn import datasets
# local
import metrics, utils
from relable import voting,relabel_cluster
from sklearn.cross_validation import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import classification_report
from simulated_annealing.optimize import SimulatedAnneal

from config import (MONGODB_HOST,
                    MONGODB_PORT,
                    MONGODB_DB,
                    MONGODB_COLLECTION,
                    KN,
                    CLUSTERING,
                    POINTS)

conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
mongodb = conn[MONGODB_DB]
collection = mongodb[MONGOD_COLLECTION]

def ForVoting():
    clusters = []
    for K in range(KN):
        data = collection.find({'KN':K})[0]
        temp = [0 for i in range(POINTS)]
        for i in range(CLUSTERING):
            for point in data['cluster_'+str(i+1)]:
                temp[int(point)-1] = i+1
        clusters.append(temp)
    relabeled_clusters = relabel_cluster(clusters)
    voting(relabeled_clusters)


if __name__=="__main__":
    create_coassoc()
    create_supers()
    ForVoting()
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    # Split the data into test and train sets                         
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # This is the hyperparameter space we'll be searching over
    svc_params = {'C':np.logspace(-8, 10, 19, base=2),
                  'fit_intercept':[True, False]
                 }
    # Using a linear SVM classifier             
    clf = svm.LinearSVC()
    # Initialize Simulated Annealing and fit
    sa = SimulatedAnneal(clf, svc_params, T=10.0, T_min=0.001, alpha=0.75,
                         verbose=True, max_iter=0.25, n_trans=5, max_runtime=300,
                         cv=3, scoring='f1_macro', refit=True)
    sa.fit(X_train, y_train)
    # Print the best score and the best params
    print(sa.best_score_, sa.best_params_)
    # Use the best estimator to predict classes
    optimized_clf = sa.best_estimator_
    y_test_pred = optimized_clf.predict(X_test)
    # Print a report of precision, recall, f1_score
    print(classification_report(y_test, y_test_pred))