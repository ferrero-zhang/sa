#-*- coding: UTF-8 -*- 

'''
1、读取指定目录下的所有文件
2、读取指定文件，输出文件内容
3、创建一个文件并保存到指定目录
'''
import os
import csv
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pymongo import MongoClient
from relable import voting,relabel_cluster

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
MONGODB_DB = 'Ifeng'
CLUSTERING_NAME = "letter"
MONGODB_COLLECTION = CLUSTERING_NAME+'_kmeans'
MONGODB_POINT = CLUSTERING_NAME+'_voting_super'
MONGODB_NA = 'hitech_na'
KN = 50 #基础聚类个数
CLUSTERING = 26  #标签数
POINTS = 20000   #数据个数
conn = MongoClient(host=MONGODB_HOST, port=MONGODB_PORT)
mongodb = conn[MONGODB_DB]
collection = mongodb[MONGODB_COLLECTION]
collection_point = mongodb[MONGODB_POINT]


# labelTXT = CLUSTERING_NAME+".txt"
# label = []

# f = open(labelTXT,'r')
# for line in f:
    # line = line.strip()
    # label.append(int(line))
# label = [1,1,2,2,3,3]
# label = np.array(label)

def writeFile(filename):
    acc = 0.0
    for count in range(1,KN+1):
        fopen = open(filename, 'r')
        temp = {}
        result = []
        clusters = []
        clusters.append(label)
        for line in fopen:
            line = line.strip().split(',')
            if temp.has_key(str(int(line[count])+1)):
                temp[str(int(line[count])+1)].append(int(line[0]))
            else:
                temp[str(int(line[count])+1)] = [int(line[0])]
            result.append(int(line[count])+1)
        fopen.close()
        print count
        clusters.append(result)
        relabeled_clusters = relabel_cluster(clusters)
        acc += accuracy_score(relabeled_clusters[0], relabeled_clusters[1])
        temp['acc'] = accuracy_score(relabeled_clusters[0], relabeled_clusters[1])
        temp['KN'] = count
        collection.insert(temp)
    print acc/KN

def cal_kmenas4HA():
    for i in range(1,KN+1):
        li = collection.find({'KN':i})[0]
        HA = 0
        for cl in range(1,CLUSTERING+1):
            HA += float(len(li[str(cl)])*(len(li[str(cl)])-1))/2.0
        li.update({'ha':HA})
        collection.update({'KN':i},li)

def countLetter():
    acc = 0.0
    for i in range(1,KN+1):
        acc += float(collection.find({'KN':i})[0]['acc'])
    print acc/KN
if __name__ == '__main__':
    filePath = "D:\\FileDemo\\Java\\myJava.txt"
    filePathI = CLUSTERING_NAME+".csv"
    filePathC = "D:\\paper\\project\\data_matrix001"
    # eachFile(filePathC)
    # readFile(filePath)
    # writeFile(filePathI)
    # cal_kmenas4HA()
    countLetter()