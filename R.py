# -*- coding: utf-8 -*-
#!/usr/bin/python

from numpy import *
import numpy as np 
import numpy
from pyspark import SparkConf, SparkContext
from pymongo import MongoClient
from pandas import Series,DataFrame
import random
import math
import time
from numpy import * 

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
MONGODB_DB = 'digit'
MONGOD_COLLECTION = 'ohscal_Kmeans'
MONGOD_COLLECTION_DATA = 'ohscal_data'
MONGOD_COLLECTION_SIM = 'ohscal_sim'
MONGOD_COLLECTION_SUP = 'ohscal_sup'
MONGOD_COLLECTION_NA = 'ohscal_na_R_1'
MONGOD_COLLECTION_R = 'ohscal_r'
MONGOD_COLLECTION_M = 'ohscal_m_R_1'
mongodb = conn[MONGODB_DB]
dataSet = []
collection_data = mongodb[MONGOD_COLLECTION_DATA]
collection_sim = mongodb[MONGOD_COLLECTION_SIM]
collection = mongodb[MONGOD_COLLECTION]
collection_sup = mongodb[MONGOD_COLLECTION_SUP] 
collection_na = mongodb[MONGOD_COLLECTION_NA]
collection_r = mongodb[MONGOD_COLLECTION_R]
collection_m = mongodb[MONGOD_COLLECTION_M]
cluster_num=10 #聚类数，K
KN = 50        #基础聚类数
N = 11162
collection_m.drop()
collection_na.drop()
listA = [i for i in range(1,cluster_num+1)]
listM = [0 for i in range(N)]
data = collection_data.find()
for li in data:
    a = random.randint(1,cluster_num+1)
    collection_data.update({'id':li['id']},{"$set":{'random_class1':a}},upsert=False)

na_data = []
for i in range(KN):
    na_data.append([])
    
temp_result = []
sim = []
for i in range(N):
    sim.append([])
    for j in range(N):
        sim[i].append(0)
#模拟退火
conf = (SparkConf()
    .set("spark.driver.maxResultSize", "16g"))
sc = SparkContext(conf=conf)
sim_matrix = []
sim_data = collection_sim.find()
for li in sim_data:
    sim_matrix.append(li)
a = sc.parallelize(sim_matrix,2000)
for line in a.collect():
    sim[int(line['row'])-1][int(line['col'])-1] = int(line['num'])

sup_pointer = 1
sup_pointers = collection_sup.find()
for sp in sup_pointers:
    if(sp['S_id']>sup_pointer):
        sup_pointer = sp['S_id']
#R策略 为每个数据样本随机分配簇标签
temp_m = collection_data.find()
m_data = []
for li in temp_m:
    m_data.append({'point':int(li['id']),'flag1':int(li['random_class1'])})
collection_m.insert(m_data,save=True)
'''
    if collection_m.find({'point':int(li['id'])}).count():
        collection_m.update({'point':int(li['id'])},{"$set":{'flag1':int(li['random_class1']),'point':int(li['id'])}},upsert=False)
    else:
        collection_m.insert({'point':int(li['id']),'flag1':int(li['random_class1'])})
'''
current = []
temp_data = []
temp_pn = []
temp_ha = []
for i in range(0,cluster_num):
    current.append([])
    temp_data.append([])
    temp_pn.append([])
    temp_ha.append([])
    for j in xrange(cluster_num):
        temp_pn[i].append([])
        temp_ha[i].append([])
datas = collection_data.find()
for li in datas:
    current[int(li['random_class1'])-1].append(int(li['id']))

ha_current = 0.0
for i in xrange(cluster_num):
    ha_current += len(current[i])*(len(current[i])-1)/2
c_r = 0.0

for K in range(0,50):
    na = 0.0
    modify_na = {}
    data1 = collection.find({'KN':K})[0]
    hb = data1['ha']
    for i in xrange(cluster_num):
        for j in xrange(cluster_num):
            temp_data[j] = data1['cluster_'+str(j+1)]
            temp_pn[i][j] = len(list(set(current[i]).intersection(set(temp_data[j]))))
            temp_ha[i][j] = temp_pn[i][j] * (temp_pn[i][j] - 1 )/2
            na += temp_ha[i][j]
            modify_na['ha'+str(i)+str(j)] = temp_ha[i][j]
            modify_na['pn'+str(i)+str(j)] = temp_pn[i][j]
    nb = ha_current - na
    nc = hb - na
    nd = N*(N-1)/2 - na - nb - nc
    r = 2*float(na+nd)/(N*(N-1))
    c_r += r
    modify_na['na'] = na
    modify_na['row'] = K
    modify_na['r'] = r
    na_data[K] = modify_na
    if collection_na.find({'row':K}).count():
        collection_na.update({'row':K},modify_na)
    else:
        collection_na.insert(modify_na)

c_r = float(c_r /50)
c_r = round(c_r,3)
print "c_r",c_r
VT = 0.99 #温度衰减值
CF0 = 0.80 #变更阈值
Iter = 10#迭代次数
t = 0.1*c_r
#t = c_r
NP = KN * 0.30
SP = KN * 0.70
RESULT = []
Flag =0
iter = 0
update_cnt = 0  #节点运动计数
result = []
times = []
for li in collection_m.find():
    listM[int(li['point'])-1] = li['flag1']
start_time = time.time()
while(iter <= Iter ):#迭代判断
    for sid in range(1,sup_pointer+1):
        spointers = collection_sup.find({'S_id':sid})
        vote = []
        s_pointers = []
        for li in spointers:
            pointer = li['data_id']
            s_pointers.append(pointer)
        # 获取当前超点所在的簇标签
        if(len(s_pointers)==1):
            rand_spointer = 0
        else:
            rand_spointer = random.randint(0,len(s_pointers)-1)
        # 获取当前超点所在的簇标签
        #print s_pointers[rand_spointer]
        
        temp_center = listM[int(s_pointers[rand_spointer])-1]

        flags = sim[int(s_pointers[rand_spointer])-1]
        NS = [] #负向投票节点
        PS = [] #正向投票节点
        for index in range(len(flags)):
            if(flags[index]>=NP):
                NS.append(int(index+1))
            else:
                PS.append(int(index+1))
        current_vote = []
        for i in xrange(cluster_num):
            current_vote.append([])
        '''
        votes = collection_m.find()
        for li in votes:
            current_vote[int(li['flag1'])-1].append(int(li['point']))
        '''
        for li in range(N):
            current_vote[listM[li]-1].append(li+1)
        NN = []
        PP = []
        Diff = []
        for i in range(0,cluster_num):
            a = len(list(set(NS).intersection(set(current_vote[i]))))
            NN.append(a)
            b = len(list(set(PS).intersection(set(current_vote[i]))))
            PP.append(b)
            Diff.append(a-b)
        #print Diff
        vote_max = sorted(Diff)[cluster_num-1]
        vote_c = Diff.index(vote_max)+1
        #print "temp_center,vote_c:",temp_center,vote_c
        #标签变化，目标值变化
        if(temp_center!=vote_c):            
            #重新计算NA
            #new_na = collection_na.find()
            new_na = na_data
            left_cluster = list(set(listA)^set([temp_center,vote_c]))
            ri_new = 0.0
            NEW_DATA = []
            for eachna in new_na:
                r_new = 0.0
                pn_new = []#初始化
                Ha_new = []
                for i in range(0,cluster_num):
                    pn_new.append([])
                    Ha_new.append([])
                    for j in range(0,cluster_num):
                        pn_new[i].append([])
                        Ha_new[i].append([])
                na_new = 0
                update_set = {}
                for i in range(0,cluster_num):
                    for j in range(0,cluster_num):
                        if (i==temp_center and j==vote_c):
                            pn_new[i][j] = eachna['pn'+str(i)+str(j)]-1
                        elif (i==vote_c and j==temp_center):
                            pn_new[i][j] = eachna['pn'+str(i)+str(j)]+1
                        else:
                            pn_new[i][j] = eachna['pn'+str(i)+str(j)]
                        Ha_new[i][j] = pn_new[i][j] * (pn_new[i][j] -1 )/2
                        na_new += Ha_new[i][j]
                        update_set['pn'+str(i)+str(j)] = pn_new[i][j]
                        update_set['ha'+str(i)+str(j)] = Ha_new[i][j]
                            
                #计算 HB
                pn_temp = collection_m.find({'flag1':int(temp_center)}).count() - 1
                pn_center = collection_m.find({'flag1':int(vote_c)}).count() + 1
                Ha_temp = pn_temp*(pn_temp - 1)/2
                Ha_center = pn_center*(pn_center-1)/2
                hb_new = Ha_temp + Ha_center 
                #print pn_temp,pn_center
                for lf in left_cluster:
                    pn_left = collection_m.find({'flag1':int(lf)}).count()
                    #print pn_left
                    Ha_left = pn_left *(pn_left -1)/2
                    hb_new += Ha_left
                nb_new = ha_current - na_new
                nc_new = hb_new - na_new
                nd_new = N*(N-1)/2 - na_new - nb_new - nc_new
                r_new = round(2*float(na_new + nd_new)/(N*(N-1)),3)
                ri_new += r_new
                update_set['r'] = r_new
                #print eachna
                id = eachna['row']
                update_set['row'] = id
                NEW_DATA.append(update_set)
            ri_new = round(float(ri_new / 50),3)  
            #print "ri_new",ri_new
            #节点运动变更
            cs = ri_new - c_r
            #print "cs",cs
            #print "t",t
            if(cs>0):
                cf = 1
            else:
                cf = math.exp(cs/t)

            
            if(cf>CF0):
                c_r = ri_new
                update_cnt += 1     
                result.append(ri_new)
                times.append(iter)
                for p in s_pointers:
                    listM[int(p)-1] = vote_c
                    #collection_m.update({'point':int(p)},{"$set":{'point':int(p),'flag1':vote_c}},upsert=False)#更新 簇标签
                    #collection_m.update({'point':str(p)},{"$set":{'point':int(p),'flag1':vote_c}},upsert=False)#更新 簇标签
                
                for i in range(len(NEW_DATA)):
                    #collection_na.update({'id':NEW_DATA[i]['id']},{'$set':NEW_DATA[i]},upsert=False)
                    na_data[NEW_DATA[i]['row']] = NEW_DATA[i]
                #print na_data,len(na_data)
                #print NEW_DATA,len(NEW_DATA)
        else:
            for p in s_pointers:
                listM[int(p)-1] = temp_center
                #collection_m.update({'point':int(p)},{"$set":{'point':int(p),'flag1':temp_center}},upsert=False)#更新 簇标签
                #collection_m.update({'point':str(p)},{"$set":{'point':int(p),'flag1':temp_center}},upsert=False)#更新 簇标签
            
    t = t * VT
    iter += 1
end_time = time.time()
cal_time = end_time - start_time
print "cal:",cal_time
#print result
#print times
for i in range(len(listM)):
    collection_m.update({'point':i+1},{'point':i+1,'flag1':listM[i]})

origin_cluster = []
new_cluster = []
origin_data = collection_data.find()
compare_data = collection_m.find()
for i in xrange(cluster_num):
    origin_cluster.append([])
    new_cluster.append([])
for li in origin_data:
    origin_cluster[li['origin_class']-1].append(li['id'])
for li in compare_data:
    new_cluster[li['flag1']-1].append(li['point'])
overlap = []
precision = []
recall = []
fmeasure = []
Fm = 0.0 
for i in xrange(cluster_num):
    overlap.append([])
    precision.append([])
    recall.append([])
    fmeasure.append([])
    for j in xrange(cluster_num):
        nij = len(list(set(origin_cluster[i]).intersection(set(new_cluster[j]))))
        overlap[i].append(nij)
        print nij,len(origin_cluster[i])
        p = float(nij)/len(origin_cluster[i])
        print "p:",p
        precision[i].append(p)
        r = float(nij)/len(new_cluster[j])
        print "r:",r
        recall[i].append(r)
        if(nij==0):
            fmeasure[i].append(0)
        else:
            fmeasure[i].append(2*p*r/(p+r))
    Fm += max(fmeasure[i])
Fm = Fm/cluster_num
temp = np.array(overlap)
#print temp
maxnum = temp.max()
kright = 0.0
while(maxnum):
    kright += temp.max()
    indices = numpy.where(temp == temp.max() )
    ind2d = zip(indices[0], indices[1])
    temp[:,ind2d[0][1]] = np.zeros(cluster_num) 
    temp[ind2d[0][0],:] = np.zeros(cluster_num)
    maxnum = temp.max()
error = 1-kright/N
print "error:",error
#print np.array(fmeasure)
print "f-measure:",Fm

fresult = open('./wine/ohscalR.jl','ab')
fresult.write('time:'+str(cal_time)+'\t')
fresult.write('error:'+str(error)+'\t')
fresult.write('f-measure:'+str(Fm)+'\n')
fresult.close()