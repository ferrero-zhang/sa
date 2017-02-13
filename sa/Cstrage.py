# -*- coding: utf-8 -*-
#!/usr/bin/python

from numpy import *
import numpy as np 
import numpy
from pymongo import MongoClient
from pandas import Series,DataFrame
import random
import math
from pyspark import SparkConf, SparkContext
from numpy import * 
import time 

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
MONGODB_DB = 'digit'
MONGOD_COLLECTION = 'ohscal_Kmeans'
MONGOD_COLLECTION_DATA = 'ohscal_data'
MONGOD_COLLECTION_SIM = 'ohscal_sim'
MONGOD_COLLECTION_SUP = 'ohscal_sup'
MONGOD_COLLECTION_NA = 'ohscal_na'
MONGOD_COLLECTION_R = 'ohscal_r'
MONGOD_COLLECTION_M = 'ohscal_m'
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
listN = [i for i in range(1,N+1)]
listM = [0 for i in range(N)]
na_data = []
for i in range(KN):
    na_data.append([])
    for j in range(KN):
        na_data[i].append([])
temp_result = []
sim = []
for i in range(N):
    sim.append([])
    for j in range(N):
        sim[i].append(0)
conf = (SparkConf()
    .set("spark.driver.maxResultSize", "32g"))
sc = SparkContext(conf=conf)
#模拟退火

#生成关联矩阵
sim_matrix = []
sim_data = collection_sim.find()
for li in sim_data:
    sim_matrix.append(li)
a = sc.parallelize(sim_matrix,5000)
for line in a.collect():
    sim[int(line['row'])-1][int(line['col'])-1] = int(line['num'])

sup_pointer = 1
sup_pointers = collection_sup.find()
for sp in sup_pointers:
    if(sp['S_id']>sup_pointer):
        sup_pointer = sp['S_id']
#R策略 为每个数据样本随机分配簇标签

#C策略 随机选择一个基础聚类

for K in range(0,50):
    na = 0
    data = collection.find({'KN':K})[0]
    temp = []
    for i in range(0,cluster_num):
        temp.append([])
        temp[i] = data['cluster_'+str(i+1)]
    ha = data['ha']
    for J in range(0,50):
        data1 = collection.find({'KN':J})[0]
        temp1 = []
        pn = []
        Ha = []
        updates_modifier = {}
        for i in range(0,cluster_num):
            temp1.append([])
            pn.append([])
            Ha.append([])
            for j in range(0,cluster_num):
                pn[i].append([])
                Ha[i].append([])
        for i in range(0,cluster_num):
            temp1[i] = data['cluster_'+str(i+1)]
        hb = data1['ha']
        na = 0
        for i in range(0,cluster_num):
            for j in range(0,cluster_num):
                pn[i][j] = len(list(set(temp[i]).intersection(set(temp[j]))))
                Ha[i][j] = pn[i][j] * (pn[i][j] - 1) /2
                na += Ha[i][j]
                updates_modifier['pn'+str(i)+str(j)] = pn[i][j]
                updates_modifier['ha'+str(i)+str(j)] = Ha[i][j]
        nb = ha - na
        nc = hb - na
        nd = N*(N-1)/2 - na - nb - nc
        r = 2*float(na + nd)/(N*(N-1))
        updates_modifier['id'] = [K,J]
        updates_modifier['row'] = K 
        updates_modifier['r'] = r
        na_data[K][J] = updates_modifier
        if collection_na.find({'id':[K,J]}).count():
            collection_na.update({'id':[K,J]},updates_modifier)
        else:
            collection_na.insert(updates_modifier)

for K in range(0,50):
    #b_r = collection_na.find({'row':K})
    b_r = na_data[K]
    bri = 0.0
    for b_ri in b_r:
        bri += b_ri['r']
    bri = round(float(bri/50),3)
    collection.update({"KN":K},{"$set":{'r':bri}},upsert=False)

randomK = random.randint(0,50)  #选取当前状态  随机
#计算Ri  设置初始T0
#collect_r = collection_na.find({'row':randomK})
collect_r = na_data[randomK]
c_r = 0.0
print randomK
for ri in collect_r:
    c_r += ri['r']

c_r = float(c_r /50)
c_r = round(c_r,3)
#print "c_r",c_r
VT = 0.99 #温度衰减值
CF0 = 0.80 #变更阈值
Iter = 0#迭代次数
t = 0.1*c_r
#t = c_r
NP = KN * 0.30
PP = KN * 0.70
RESULT = []
spointers = collection_sup.find()
#迭代退火过程
#i 投票过程

#选取当前状态
current_data = collection.find({'KN':randomK})[0]
current = []
m_data = []
for i in range(0,cluster_num):
    current.append([])
    current[i] = current_data['cluster_'+str(i+1)]
    for li in current[i]:
        m_data.append({'point':int(li),'flag1':int(i+1)})
collection_m.insert(m_data,save=True)
'''
        if collection_m.find({'point':current}).count():
            collection_m.update({'point':int(li)},{'flag1':int(i+1),'point':int(li)})
        else:
            collection_m.insert({'point':int(li),'flag1':int(i+1)})
'''
ha_current = current_data['ha']
Flag =0
iter = 0
update_cnt = 0  #节点运动计数
result = []
times = []
for li in collection_m.find():
    listM[int(li['point'])-1] = li['flag1']
print "c_r",c_r
start_time = time.time()

while(iter <= Iter):#迭代判断
    
    for sid in range(1,sup_pointer+1):
        spointers = collection_sup.find({'S_id':sid})
        vote = []
        s_pointers = []
        for li in spointers:
            pointer = li['data_id']
            s_pointers.append(pointer)
        # 获取当前超点所在的簇标签
        #print len(s_pointers)
        if(len(s_pointers)==1):
            rand_spointer = 0
        else:
            rand_spointer = random.randint(0,len(s_pointers)-1)
        # 获取当前超点所在的簇标签
        #print s_pointers[rand_spointer]
        
        temp_center = listM[s_pointers[rand_spointer]-1]
        
        flags = sim[int(s_pointers[rand_spointer])-1]
        #flags = collection_sim.find({'row':int(s_pointers[rand_spointer])})
        NS = [] #负向投票节点
        PS = [] #正向投票节点
        '''
        for flag in flags:
            if(flag['num']>=NP):
                NS.append(int(flag['col']))
            else:
                PS.append(int(flag['col']))
        '''
        for index in range(len(flags)):
            if(flags[index]>=NP):
                NS.append(int(index+1))
            else:
                PS.append(int(index+1))
        
        #left_point = list(set(listN)^set(NS))
        #for li in left_point:
            #PS.append(int(li))
        NN = []
        PP = []
        Diff = []
        for i in range(0,cluster_num):
            #print NS
            #print current[i]
            a = len(list(set(NS).intersection(set(current[i]))))
            #print a
            NN.append(a)
            b = len(list(set(PS).intersection(set(current[i]))))
            #print b
            PP.append(b)
            Diff.append(a-b)
        #print Diff
        vote_max = sorted(Diff)[cluster_num-1]
        vote_c = Diff.index(vote_max)+1
        #print "temp_center:",temp_center
        #print "vote_c:",vote_c
        #标签变化，目标值变化
        #PN = collection_na.find({'row':randomK})
        if(temp_center!=vote_c):            
            #重新计算NA
            #new_na = collection_na.find({'row':randomK})
            left_cluster = list(set(listA)^set([temp_center,vote_c]))
            ri_new = 0.0
            NEW_DATA = []
            for eachna in collect_r:
                #eachna = eachna[0]
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

                #计算 HB
                pn_temp = current_data['pn_'+str(temp_center)] - 1
                pn_center = current_data['pn_'+str(vote_c)] + 1
                Ha_temp = pn_temp*(pn_temp - 1)/2
                Ha_center = pn_center*(pn_center-1)/2
                hb_new = Ha_temp + Ha_center 
                for lf in left_cluster:
                    Ha_left = current_data['ha_'+str(lf)]
                    hb_new += Ha_left
                nb_new = ha_current - na_new
                nc_new = hb_new - na_new
                nd_new = N*(N-1)/2 - na_new - nb_new - nc_new
                r_new = round(2*float(na_new + nd_new)/(N*(N-1)),3)
                update_set['r'] = r_new
                id = eachna['id']
                update_set['id'] = id
                ri_new += r_new
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
                #for i in range(len(NEW_DATA)):
                    #collection_na.update({'id':NEW_DATA[i]['id']},{'$set':NEW_DATA[i]},upsert=False)
                    #na_data[NEW_DATA[i]['row']] = NEW_DATA[i]
            
        else:
            for p in s_pointers:
                listM[int(p)-1] = temp_center
                #collection_m.update({'point':int(p)},{"$set":{'point':int(p),'flag1':temp_center}},upsert=False)#更新 簇标签
                #collection_m.update({'point':str(p)},{"$set":{'point':int(p),'flag1':temp_center}},upsert=False)#更新 簇标签
            
    t = t * VT
    iter += 1
end_time = time.time()
cal_time = end_time - start_time
#print result
#print times
print "cal:",cal_time
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
for i in xrange(cluster_num):
    overlap.append([])
    for j in xrange(cluster_num):
        overlap[i].append(len(list(set(origin_cluster[i]).intersection(set(new_cluster[j])))))
temp = np.array(overlap)
maxnum = temp.max()
kright = 0.0
while(maxnum):
    kright += temp.max()
    indices = numpy.where(temp == temp.max() )
    ind2d = zip(indices[0], indices[1])
    temp[:,ind2d[0][1]] = np.zeros(cluster_num) 
    temp[ind2d[0][0],:] = np.zeros(cluster_num)
    maxnum = temp.max()
print "error:",1-kright/N
fresult = open('ohscal.jl','ab')
fresult.write('time:'+str(cal_time)+'\n')
fresult.write('error:'+str(1-kright/N)+'\n')
fresult.close()
