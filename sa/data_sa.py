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
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics.cluster import normalized_mutual_info_score
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
from multiprocessing import Process,Lock
from multiprocessing import Process, Value, Array
import logging
from multiprocessing import Pool
from json2csv import MainJson2Csv
import sys
import ConfigParser

from config import (MONGODB_HOST,
                    MONGODB_PORT,
                    MONGODB_DB,
                    MONGODB_COLLECTION,
                    KN,
                    CLUSTERING,
                    MONGODB_POINT,
                    MONGODB_NA,
                    CLUSTERING_NAME,
                    # RLABLE,
                    POINTS)
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='/home/ubuntu7/zhangzhibin/mnist/sa/sa/sa_'+CLUSTERING_NAME+'.log',
                filemode='a')
conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
mongodb = conn[MONGODB_DB]
collection = mongodb[MONGODB_COLLECTION]
collection_point = mongodb[MONGODB_POINT]
collection_na = mongodb[MONGODB_NA]

points =[i for i in range(1,POINTS+1)]
global CKN,VOTE,T,alpha,max_iter,cf,T_min,NADATA
config = ConfigParser .ConfigParser()
config.read('/home/ubuntu7/zhangzhibin/mnist/sa/sa/params.config')
CKN = config.get('params','KN')
T = float(config.get('params','T'))
all_count = int(config.get('params','all_count'))
# CKN = 32
# T = 0.0995428956948
alpha = 0.9
max_iter = 10
cf = 0.8
T_min = 0.000001
VOTE = pd.read_csv("/home/ubuntu7/zhangzhibin/mnist/sa/sa/vote_super.csv")
NADATA = pd.read_csv("/home/ubuntu7/zhangzhibin/mnist/sa/sa/na.csv")
RLABLE = [0 for i in range(POINTS)]
def Rfit(SUPER,RLABLE):
    global CKN,VOTE,T,alpha,max_iter,cf,T_min,NADATA
    # Set up  the initial params
    # Computes the acceptance probability as a function of T; maximization
    accept_prob = lambda old, new, T: np.exp((new-old)/T)
    total_iter = 0
    old_score = T*10
    ST = time.time()
    # print(VOTE)
    while T > T_min and total_iter < max_iter :

        if len(SUPER)==1:
            voteNum = SUPER[0]-1
        else:
            voteNum = SUPER[random.randint(1,len(SUPER))-1]-1
        if isinstance(voteNum,list):
            voteNum = voteNum[0]
        origin_vote = VOTE.ix[[voteNum],['random_label']].values.tolist()[0][0]
        new_vote = VOTE.ix[[voteNum],['vote']].values.tolist()[0][0]
        # if(np.isnan(new_vote)):
            # print("@"*50)
        if new_vote != origin_vote and not np.isnan(new_vote) and not np.isnan(origin_vote):
            '''计算 HB,r......
            '''
            # print(new_vote,origin_vote)
            for i in range(1,CLUSTERING+1):
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
                    '''VOTE.ix[[point-1],['random_label']] = new_vote
                    '''
                    if isinstance(point,list):
                        for p in point:
                            RLABLE[p-1] = int(new_vote)
                    else:
                        RLABLE[point-1] = int(new_vote)
                # print(new_vote)
            else:
                for point in SUPER:
                    '''VOTE.ix[[point-1],['random_label']] = origin_vote
                    '''
                    if isinstance(point,list):
                        for p in point:
                            RLABLE[p-1] = int(new_vote)
                    else:
                        RLABLE[point-1] = int(origin_vote)
                    # print(origin_vote)
        T *= alpha
        total_iter += 1

def Rfit2(super,RLABLE):
    global CKN,VOTE,T,alpha,max_iter,cf,T_min,NADATA
    # Set up  the initial params
    # Computes the acceptance probability as a function of T; maximization
    accept_prob = lambda old, new, T: np.exp((new-old)/T)
    total_iter = 0
    old_score = T*10
    ST = time.time()
    # print(VOTE)
    while T > T_min and total_iter < max_iter :
        for SUPER in super:
            if len(SUPER)==1:
                voteNum = SUPER[0]-1
            else:
                voteNum = SUPER[random.randint(1,len(SUPER))-1]-1
            if isinstance(voteNum,list):
                voteNum = voteNum[0]-1
            
            origin_vote = VOTE.ix[[voteNum],['random_label']].values.tolist()[0][0]
            new_vote = VOTE.ix[[voteNum],['vote']].values.tolist()[0][0]
            if(np.isnan(new_vote)):
                print(voteNum)
            if new_vote != origin_vote:
                '''计算 HB,r......
                '''
                # print(new_vote,origin_vote)
                for i in range(1,CLUSTERING+1):
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
                        '''VOTE.ix[[point-1],['random_label']] = new_vote
                        '''
                        if isinstance(point,list):
                            for p in point:
                                RLABLE[p-1] = int(new_vote)
                        else:
                            RLABLE[point-1] = int(new_vote)
                    # print(new_vote)
                else:
                    for point in SUPER:
                        '''VOTE.ix[[point-1],['random_label']] = origin_vote
                        '''
                        if isinstance(point,list):
                            for p in point:
                                RLABLE[p-1] = int(new_vote)
                        else:
                            RLABLE[point-1] = int(origin_vote)
                        # print(origin_vote)
        T *= alpha
        total_iter += 1

def Cfit(SUPER):
    global CKN,VOTE,T,alpha,max_iter,cf,T_min,NADATA
    # Set up  the initial params
    # Computes the acceptance probability as a function of T; maximization
    accept_prob = lambda old, new, T: np.exp((new-old)/T)
    total_iter = 10
    old_score = T*10
    ST = time.time()
    while T > T_min and total_iter < max_iter :
        # print(SUPER)
        if len(SUPER)==1:
            voteNum = SUPER[0]-1
        else:
            voteNum = SUPER[random.randint(1,len(SUPER))-1]-1

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
    '''获取超点集合
    '''
    SUPER = []
    for sid in range(1,all_count):
        spointers = collection_point.find({'S_id':sid})
        s_pointers = []
        for li in spointers:
            pointer = li['data_id']
            s_pointers.append(pointer)
        if len(s_pointers)!=0:
            SUPER.append(s_pointers)
    label = []      # 真实标签
    f = open('/home/ubuntu7/zhangzhibin/mnist/'+CLUSTERING_NAME+'.txt')
    for line in f:
        line = line.strip()
        label.append(int(line))
    # label = np.array(label)
    # print(VOTE)
    startTime = time.time()
    pool = Pool()
    if sys.argv[1]=="C":
        pool.map(Cfit,SUPER)
        pool.close()
        pool.join()
        endTime = time.time()
        cal = endTime - startTime
        print("cal time is :",cal)
        predict_result = VOTE['origin_vote'].values.tolist() 
        
    else:
        # num = Value('d', 0.0)
        tarr = [0 for i in range(POINTS-1)]
        num = Value('d', 0.0)
        arr = Array('i', [1]*POINTS)
        ps = []
        for sup in SUPER:
            ps.append(Process(target=Rfit, args=(sup, arr)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()
        # Rfit2(SUPER,arr)
        # p.start()
        # p.join()
        # pool.map(p)
        # pool.close()
        # pool.join()
        # Rfit(SUPER)
        endTime = time.time()
        cal = endTime - startTime
        print("cal time is :",cal)
        # predict_result = VOTE['random_label'].values 
        predict_result = arr[:]
        # predict_result = []
        # f = open('result.txt','w')
        # for i in arr[:]:
            # f.write(str(i)+'\n')
        # f.close()
        # f = open('result.txt')
        # for line in f:
            # line = int(line.strip())
            # predict_result.append(line)
        # print(predict_result)
        # print(VOTE)
        # clusters = []
        # clusters.append(label)
        # clusters.append(predict_result)
        # relabeled_clusters = relabel_cluster(clusters)
        # logging.info("for the strategy:"+sys.argv[1]) 
        # logging.info("KN:"+str(CKN)+'\t'+str(cal)) 
        # logging.info('\n f1_score:'+f1_score(relabeled_clusters[0], relabeled_clusters[1],average='macro')) 
        # logging.info('\n accuracy_score:'+accuracy_score(relabeled_clusters[0], relabeled_clusters[1])) 
        # logging.info('\n recall_score'+recall_score(relabeled_clusters[0], relabeled_clusters[1],average='macro')) 
        # Print a report of precision, recall, f1_score
        # print(classification_report(np.array(relabeled_clusters[0]), np.array(relabeled_clusters[1])))
    clusters = []
    clusters.append(label)
    clusters.append(predict_result)
    
    relabeled_clusters = relabel_cluster(clusters)
    logging.info("for the strategy:"+sys.argv[1]) 
    logging.info("KN:"+str(CKN)+'\t'+str(cal)) 
    # print(label)
    # print(predict_result)
    # logging.info('\n accuracy_score:'+accuracy_score(relabeled_clusters[0], relabeled_clusters[1])) 
    logging.info('\n'+classification_report(np.array(relabeled_clusters[0]), np.array(relabeled_clusters[1]))) 
    # logging.info('\n'+normalized_mutual_info_score(np.array(relabeled_clusters[0]), np.array(relabeled_clusters[1]))) 
    # Print a report of precision, recall, f1_score
    print(classification_report(np.array(relabeled_clusters[0]), np.array(relabeled_clusters[1])))
    

      