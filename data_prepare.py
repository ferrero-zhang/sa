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
config.read('D:/github/sa/params.config')

def pre_data():
    #for relabel & voting 
    f = open('D:/github/sa/vote_super.json','w')
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
    # collection_point.insert(result)
    for li_result in result:
        f.write(json.dumps(li_result))
        f.write('\n')
    # print(s_id)
    f.close()
    MainJson2Csv('vote_super')
    """insert super_points data table
    """
    return s_id
def cal_kmenas4HA():
    for i in range(1,KN+1):
        li = collection.find({'KN':i})[0]
        HA = 0
        for cl in range(1,CLUSTERING+1):
            HA += float(len(li[str(cl)])*(len(li[str(cl)])-1))/2.0
        li.update({'ha':HA})
        collection.update({'KN':i},li)

def initNa(CKN):
    f = open('D:/github/sa/na.json','w')
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

def fit(SUPER):
    global VOTE
    # Set up  the initial params
    T = 0.0995
    alpha = 0.9
    max_iter = 5
    cf = 0.8
    T_min = 0.000001
    # Computes the acceptance probability as a function of T; maximization
    accept_prob = lambda old, new, T: np.exp((new-old)/T)
    total_iter = 0
    # print("test...................")
    '''引入投票结果
    '''
    VOTE = pd.read_csv("D:/github/sa/vote_super.csv")
    # VOTE = sqlContext.read.json("D:/github/sa/vote_super.json")
    # VOTE = VOTE.toPandas()
    '''引入NA
    '''
    NADATA = pd.read_csv("D:/github/sa/na.csv")
    # NADATA = sqlContext.read.json("D:/github/sa/na.json")
    # NADATA = NADATA.toPandas()
    old_score = T*10
    ST = time.time()
    while T > T_min and total_iter < max_iter :
        # print(SUPER)
        if len(SUPER)==1:
            voteNum = SUPER[0]
        else:
            voteNum = SUPER[random.randint(1,len(SUPER))-1]
        # print(voteNum)
        origin_vote = VOTE.ix[[voteNum],['origin_vote']].values.tolist()[0][0]
        new_vote = VOTE.ix[[voteNum],['vote']].values.tolist()[0][0]
        # print(origin_vote,new_vote)
        if new_vote != origin_vote:
            '''计算 HB,r......
            '''
            for i in range(1,CLUSTERING+1):
                # print(origin_vote,new_vote,i)
                NADATA[str(new_vote)+str(i)] += 1
                NADATA[str(origin_vote)+str(i)] -= 1
            NADATA['na']=0
            for i in range(1,CLUSTERING+1):
                for j in range(1,CLUSTERING+1):
                    NADATA['na'] += NADATA[str(i)+str(j)]*(NADATA[str(i)+str(j)]-1)/2
                    
            NADATA['nb_new'] = NADATA['ha'] - NADATA['na']
            NADATA['nc_new'] = NADATA['hb'] - NADATA['na']
            NADATA['nd_new'] = POINTS* (POINTS-1)/2 - NADATA['na'] - NADATA['nb_new'] - NADATA['nc_new']
            NADATA['r'] = 2*(NADATA['na']+NADATA['nd_new'])/(POINTS*(POINTS-1)/2)
            new_score = NADATA['r'].sum()/KN  
            old_score = old_score
                
            if new_score - old_score >0:
                a = 1
            else:
                a = accept_prob(old_score, new_score, T)
            if a > cf:
                old_score = new_score
                '''update data 
                '''
                for point in SUPER:
                    VOTE.ix[[point-1],['origin_vote']] = new_vote
            else:
                for point in SUPER:
                    VOTE.ix[[point-1],['origin_vote']] = origin_vote
            # for point in li:
                # VOTE.ix[[point-1],['origin_vote']] = new_vote
        T *= alpha
        total_iter += 1
    ET = time.time()
    # print("="*20,ET-ST)
    # return VOTE['origin_vote'].values
    # print(VOTE['origin_vote'].values)  


     
if __name__=="__main__":
    Scount = pre_data()  
    print("all_count:",Scount)
    # CKN = random.randint(1,KN)
    # Initialize Simulated Annealing 
    InitT,ha = initNa(CKN) # 初始温度
    print("InitT",InitT)

    config.set('params','KN',CKN)
    config.set('params','T',InitT)
    config.write(open('D:/github/sa/params.config', 'w'))
    

      