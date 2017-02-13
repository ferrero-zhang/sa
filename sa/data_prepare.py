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
import pandas as pd
from pandas import Series,DataFrame
import random 
from collections import Counter
import json
import time
from multiprocessing import Process
import logging
from multiprocessing import Pool
from json2csv import MainJson2Csv
import ConfigParser


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

points =[i for i in range(1,POINTS+1)]
global CKN,VOTE
CKN = random.randint(1,KN)
print("random KN:",CKN)
config = ConfigParser.ConfigParser()
config.read('/home/ubuntu7/zhangzhibin/mnist/sa/sa/params.config')

def pre_data():
    #for relabel & voting 
    f = open('/home/ubuntu7/zhangzhibin/mnist/sa/sa/vote_super.json','w')
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
            result[each-1].update({'S_id':s_id,'data_cnt':len(sup)})
        count += len(sup)
        s_id += 1
    left_point = list(set(points)^set(super_pointers))
    for each in left_point:
        result[each-1].update({'S_id':s_id,'data_cnt':1}) 
        s_id += 1
    Cvote = collection.find({'KN':CKN})[0]
    for i in range(1,CLUSTERING+1):
        for p in Cvote[str(i)]:
            result[p-1].update({'origin_vote':i,"random_label":random.randint(1,CLUSTERING)})
    # print(len(super_pointers))
    #collection_point.insert(result)
    for li_result in result:
        f.write(json.dumps(li_result))
        # collection_point.insert(li_result)
        f.write('\n')
    # print(s_id)
    f.close()
    MainJson2Csv('vote_super')
    """insert super_points data table
    """
    return s_id


def initNa(CKN):
    f = open('/home/ubuntu7/zhangzhibin/mnist/sa/sa/na.json','w')
    base = collection.find({'KN':CKN})[0]
    tempR = 0
    ha = base['ha']
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

        hb = li['ha']
        nb = ha - na
        nc = hb - na
        nd = POINTS * (POINTS-1)/2 - na - nb - nc
        r = 2*float(na + nd)/(POINTS*(POINTS-1))
        tempNa['r'] = r
        tempNa['hb'] = hb
        tempNa['ha'] = ha
        # collection_na.insert(tempNa)
        tempR += r
        # print(tempNa['hb'])
        f.write(json.dumps(tempNa))
        f.write('\n')   
    f.close()
    MainJson2Csv('na')
    return 0.1*tempR/KN,ha

     
if __name__=="__main__":
    Scount = pre_data()  
    print("all_count:",Scount)
    # CKN = random.randint(1,KN)
    # Initialize Simulated Annealing 
    InitT,ha = initNa(CKN) # 初始温度
    print("InitT",InitT)

    config.set('params','KN',CKN)
    config.set('params','all_count',Scount)
    config.set('params','T',InitT)
    config.write(open('/home/ubuntu7/zhangzhibin/mnist/sa/sa/params.config', 'w'))
    

      