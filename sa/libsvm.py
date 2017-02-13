# -*- coding: utf-8 -*-

import sys
import os
path = 'D:\libsvm-3.21\python'
sys.path.append(path)
os.chdir(path)
from svmutil import *
import numpy as np
from sklearn.metrics import classification_report
import logging
from pymongo import MongoClient
import random

from config import (MONGODB_HOST,
                    MONGODB_PORT,
                    MONGODB_DB,
                    MONGODB_COLLECTION,
                    KN,
                    CLUSTERING,
                    POINTS)
conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
mongodb = conn[MONGODB_DB]
collection = mongodb[MONGODB_COLLECTION]
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='E:\libsvm_MINIST.log',
                filemode='a')

def baseClustering():
    y, x = svm_read_problem('D:\libsvm-3.21\dataset\mnist')
    
    target = np.array(y)
    for i in range(100):
        para = '-c '+str(random.randint(1,20))
        m = svm_train(y, x,para)
        clusters = {}
        p_label, p_acc, p_val = svm_predict(y, x, m )
        for point in range(1,int(POINTS)+1):
            if clusters.has_key(str(int(p_label[point-1]))):
                clusters[str(int(p_label[point-1]))].append(point)
            else:
                clusters[str(int(p_label[point-1]))] = [point]
        clusters['error'] = 1- p_acc[0]/100
        clusters['acc'] = p_acc[0]/100
        clusters['KN'] = i+1
        collection.insert(clusters)
        p_label = np.array(p_label)
        logging.info(str(i))
        logging.info(classification_report(target, p_label))

if __name__=="__main__":
    # print(classification_report(y_test, y_test_pred))
    baseClustering()