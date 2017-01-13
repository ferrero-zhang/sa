# -*- coding: utf-8 -*-


# Imports
# compat
from __future__ import absolute_import, division, print_function
from six.moves import map, range, zip
import six
from pymongo import MongoClient
# 3rd party
import numpy as np
from sklearn.metrics import classification_report
# local
import metrics, utils
from relable import voting,relabel_cluster
from co_association import co_association
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

def pre_data():
    #for relabel & voting 
    clusters = []
    for K in range(KN):
        data = collection.find({'KN':K})[0]
        temp = [0 for i in range(POINTS)]
        for i in range(CLUSTERING):
            for point in data['cluster_'+str(i+1)]:
                temp[int(point)-1] = i+1
        clusters.append(temp)
    relabeled_clusters = relabel_cluster(clusters) #对齐标签
    voting(relabeled_clusters)                     #投票
    """insert voting data table
    """
    # count super_pointers
    count = 0
    relabel_dict = {}
    for li in relabeled_clusters:
        for point in range(1,POINTS+1):
            if relabel_dict.has_key(li[point-1]):
                relabel_dict[li[point-1]].append(point)
            else:
                relabel_dict[li[point-1]] = [point]
    for i in range(1,CLUSTERING+1):
        co_ = Counter(relabel_dict[i])
        sup = []
        for k,v in co_.items():
            if v == KN:
                sup.append(k)
        count += len(sup)
    """insert super_points data table
    """

def initTemp():
    
if __name__=="__main__":
    pre_data()
    # Initialize Simulated Annealing 
    # InitT = 0.46 # 初始温度
    sa = SimulatedAnneal(T=initTemp(), max_iter=10, alpha=0.75,cf=0.8)
    label = []      # 真实标签
    predict_result = sa.fix() # 预测标签
    # Print a report of precision, recall, f1_score
    print(classification_report(label, predict_result))