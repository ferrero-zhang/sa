# -*- coding: utf-8 -*-

import sys
import time
from copy import copy
import random
import sklearn.cross_validation as cross_validation
from sklearn.base import clone
import numpy as np
import pandas as pd 
from sklearn.metrics.scorer import get_scorer
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import _fit_and_score
from pymongo import MongoClient
from pyspark import HiveContext
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark import *  
import time


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
conf = SparkConf().setMaster("local").set("spark.executor.memory", "512m").setAppName("SparkSQL").set("spark.eventLog.enabled",False)
sc = SparkContext(conf=conf)
sqlContext = HiveContext(sc)
na = sqlContext.read.json("D:/github/sa/na.json")
class SimulatedAnneal(object):
    def __init__(self, T=0.5, max_iter=10,T_min=0.0001, alpha=0.75,cf=0.8):

        assert alpha <= 1.0
        assert T > T_min
        assert max_iter is not None and max_iter > 0
        self.__T = T
        self.__alpha = alpha
        self.__max_iter = max_iter
        self.__cf = cf
        self.__T_min = T_min

    def fit(self,SUPER,ha):
        # Set up  the initial params
        T = self.__T
        alpha = self.__alpha
        max_iter = self.__max_iter
        cf = self.__cf
        T_min = self.__T_min
        # Computes the acceptance probability as a function of T; maximization
        accept_prob = lambda old, new, T: np.exp((new-old)/T)
        total_iter = 0
        print("test...................")
        '''引入投票结果
        '''
        VOTE = sqlContext.read.json("D:/github/sa/vote_super.json")
        VOTE = VOTE.toPandas()
        '''引入NA
        '''
        NADATA = sqlContext.read.json("D:/github/sa/na.json")
        NADATA = NADATA.toPandas()
        old_score = T*10
        ST = time.time()
        tempIter = False
        while T > T_min and total_iter < max_iter :
            
            for li in SUPER:
                voteNum = li[random.randint(1,len(li))-1]
                origin_vote = VOTE.ix[[voteNum],['origin_vote']].values.tolist()[0][0]+1
                new_vote = VOTE.ix[[voteNum],['vote']].values.tolist()[0][0]
                # print(origin_vote,new_vote)
                if new_vote != origin_vote:
                    '''计算 HB,r......
                    '''
                    for i in range(1,CLUSTERING+1):
                        NADATA[str(new_vote)+str(i)] += 1
                        NADATA[str(origin_vote)+str(i)] -= 1
                    NADATA['na']=0
                    for i in range(1,CLUSTERING+1):
                        for j in range(1,CLUSTERING+1):
                            NADATA['na'] += NADATA[str(i)+str(j)]*(NADATA[str(i)+str(j)]-1)/2
                    
                    NADATA['nb_new'] = ha - NADATA['na']
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
                        for point in li:
                           VOTE.ix[[point-1],['origin_vote']] = new_vote
                    else:
                        for point in li:
                            VOTE.ix[[point-1],['origin_vote']] = origin_vote
                    # for point in li:
                        # VOTE.ix[[point-1],['origin_vote']] = new_vote
            T *= alpha
            total_iter += 1
        ET = time.time()
        print("="*20,ET-ST)
        return VOTE['origin_vote'].values
        # print(VOTE['origin_vote'].values)
        
if __name__=="__main__":
    sa = SimulatedAnneal(T=0.0995, max_iter=10, alpha=0.9,cf=0.8)
    SUPER = []
    for sid in range(1,2589):
        spointers = collection_point.find({'S_id':sid})
        s_pointers = []
        for li in spointers:
            pointer = li['data_id']
            s_pointers.append(pointer)
        SUPER.append(s_pointers)
    sa.fit(SUPER)
