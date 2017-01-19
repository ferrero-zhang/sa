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
from pyspark import HiveContext
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark import *
import pandas as pd
from pandas import Series,DataFrame
import random 
from collections import Counter
import json
import time
# SparkSession.builder.config(conf=SparkConf())
from config import (MONGODB_HOST,
                    MONGODB_PORT,
                    MONGODB_DB,
                    MONGODB_COLLECTION,
                    KN,
                    CLUSTERING,
                    MONGODB_POINT,
                    MONGODB_NA,
                    POINTS)

conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
mongodb = conn[MONGODB_DB]
collection = mongodb[MONGODB_COLLECTION]
collection_point = mongodb[MONGODB_POINT]
collection_na = mongodb[MONGODB_NA]
# conf = SparkConf().setMaster("local").set("spark.executor.memory", "512m").setAppName("SparkSQL")
# sc = SparkContext(conf=conf)
# sqlContext = HiveContext(sc)
points =[i for i in range(1,POINTS+1)]
def pre_data():
    #for relabel & voting 
    f = open('vote_super.json','w')
    result = []
    clusters = []
    origin_vote = {}
    for K in range(1,KN+1):
        data = collection.find({'KN':K})[0]
        temp = [0 for i in range(POINTS)]
        for i in range(CLUSTERING):
            for point in data[str(i+1)]:
                temp[int(point)-1] = i+1
                origin_vote[int(point)] = i
        clusters.append(temp)
    relabeled_clusters = relabel_cluster(clusters) #对齐标签
    vote_result = voting(relabeled_clusters)                #投票
    """insert voting data table
    """
    for i in range(len(vote_result)):
        result.append({'data_id':i+1,'vote':vote_result[i]})
    # count super_pointers
    count = 0
    relabel_dict = {}
    for li in relabeled_clusters:
        for point in range(1,POINTS+1):
            if relabel_dict.has_key(li[point-1]):
                relabel_dict[li[point-1]].append(point)
            else:
                relabel_dict[li[point-1]] = [point]
    s_id = 1
    super_pointers = []
    for i in range(1,CLUSTERING+1):
        co_ = Counter(relabel_dict[i])
        sup = []
        for k,v in co_.items():
            if v == KN:
                sup.append(k)
        super_pointers.extend(sup)
        for each in sup:
            result[each-1].update({'S_id':s_id,'data_cnt':len(sup),'origin_vote':origin_vote[each]})
        count += len(sup)
        s_id += 1
    left_point = list(set(points)^set(super_pointers))
    for each in left_point:
        result[each-1].update({'S_id':s_id,'data_cnt':1,'origin_vote':origin_vote[each]}) 
        s_id += 1
    # print(len(super_pointers))
    # collection_point.insert(result)
    for li_result in result:
        f.write(json.dumps(li_result))
        f.write('\n')
    print(s_id)
    f.close()
    """insert super_points data table
    """
def cal_kmenas4HA():
    for i in range(1,KN+1):
        li = collection.find({'KN':i})[0]
        HA = 0
        for cl in range(1,CLUSTERING+1):
            HA += float(len(li[str(cl)])*(len(li[str(cl)])-1))/2.0
        li.update({'ha':HA})
        collection.update({'KN':i},li)

def initNa(CKN):
    f = open('na.json','w')
    base = collection.find({'KN':CKN})[0]
    tempR = 0
    for i in range(1,KN+1):
        na = 0
        tempNa = {}
        li = collection.find({'KN':i})[0]
        for cli in range(1,CLUSTERING+1):
            for clj in range(1,CLUSTERING+1):
                pn = len(list(set(base[str(cli)]).intersection(set(li[str(clj)]))))
                tempNa[str(cli)+str(clj)] = pn
                na += pn*(pn-1)/2
        tempNa['na'] = na
        ha = base['ha']
        hb = li['ha']
        nb = ha - na
        nc = hb - na
        nd = POINTS * (POINTS-1)/2 - na - nb - nc
        r = 2*float(na + nd)/(POINTS*(POINTS-1))
        tempNa['r'] = r
        # collection_na.insert(tempNa)
        tempR += r
        f.write(json.dumps(tempNa))
        f.write('\n')   
    f.close()
    return 0.1*tempR/KN
                
        
if __name__=="__main__":
    # pre_data()
    CKN = random.randint(1,KN)
    # Initialize Simulated Annealing 
    InitT = initNa(CKN) # 初始温度
    # print(InitT)
    sa = SimulatedAnneal(T=InitT, max_iter=10, alpha=0.9,cf=0.8)
    startTime = time.time()
    predict_result = sa.fit() # 组合聚类生产最后标签
    endTime = time.time()
    cal = endTime - startTime
    print("cal time is :",cal)
    label = []      # 真实标签
    f = open('D:/github/sa/letter.txt')
    for line in f:
        line = line.strip()
        label.append(int(line))
    label = np.array(label)
    
    # Print a report of precision, recall, f1_score
    print(classification_report(label, predict_result))
    

      