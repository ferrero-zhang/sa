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
import sys
from config import (MONGODB_HOST,
                    MONGODB_PORT,
                    MONGODB_DB,
                    MONGODB_COLLECTION,
                    KN,
                    CLUSTERING,
                    MONGODB_POINT,
                    MONGODB_NA,
                    POINTS)
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='D:/github/sa/sa_letter.log',
                filemode='a')
conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
mongodb = conn[MONGODB_DB]
collection = mongodb[MONGODB_COLLECTION]
collection_point = mongodb[MONGODB_POINT]
collection_na = mongodb[MONGODB_NA]

points =[i for i in range(1,POINTS+1)]
global CKN,VOTE,T,alpha,max_iter,cf,T_min
CKN = 32
T = 0.0995428956948
alpha = 0.9
max_iter = 10
cf = 0.8
T_min = 0.000001
VOTE = pd.read_csv("D:/github/sa/vote_super.csv")


def Rfit(SUPER):
    global CKN,VOTE,T,alpha,max_iter,cf,T_min
    # Set up  the initial params
    # Computes the acceptance probability as a function of T; maximization
    accept_prob = lambda old, new, T: np.exp((new-old)/T)
    total_iter = 0
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
        origin_vote = VOTE.ix[[voteNum],['random_label']].values.tolist()[0][0]
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
                    VOTE.ix[[point-1],['random_label']] = new_vote
            else:
                for point in SUPER:
                    VOTE.ix[[point-1],['random_label']] = origin_vote
            # for point in li:
                # VOTE.ix[[point-1],['origin_vote']] = new_vote
        T *= alpha
        total_iter += 1
    ET = time.time()


def Cfit(SUPER):
    global CKN,VOTE,T,alpha,max_iter,cf,T_min
    # Set up  the initial params
    # Computes the acceptance probability as a function of T; maximization
    accept_prob = lambda old, new, T: np.exp((new-old)/T)
    total_iter = 0
    # print("test...................")
    '''引入投票结果
    '''
    # VOTE = pd.read_csv("D:/github/sa/vote_super.csv")
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
    # Scount = pre_data()  #2589
    # CKN = random.randint(1,KN)
    # Initialize Simulated Annealing 
    # InitT,ha = initNa(CKN) # 初始温度
    # print(InitT)
    # sa = SimulatedAnneal(T=InitT, max_iter=10, alpha=0.9,cf=0.8)
    '''获取超点集合
    '''
    SUPER = []
    for sid in range(1,2589):
        spointers = collection_point.find({'S_id':sid})
        s_pointers = []
        for li in spointers:
            pointer = li['data_id']
            s_pointers.append(pointer)
        SUPER.append(s_pointers)
    label = []      # 真实标签
    f = open('D:/github/sa/letter.txt')
    for line in f:
        line = line.strip()
        label.append(int(line))
    label = np.array(label)
    
    startTime = time.time()
    pool = Pool()
    if sys.argv[1]=="C":
        pool.map(Cfit,SUPER)
        pool.close()
        pool.join()
        endTime = time.time()
        cal = endTime - startTime
        print("cal time is :",cal)
        predict_result = VOTE['origin_vote'].values 
    else:
        pool.map(Rfit,SUPER)
        pool.close()
        pool.join()
        endTime = time.time()
        cal = endTime - startTime
        print("cal time is :",cal)
        predict_result = VOTE['random_label'].values 
    logging.info("for the strategy:"+sys.argv[1]) 
    logging.info("KN:"+str(CKN)+'\t'+str(cal)) 
    logging.info('\n'+classification_report(label, predict_result)) 
    # Print a report of precision, recall, f1_score
    print(classification_report(label, predict_result))

    

      