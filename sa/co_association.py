# -*- coding: utf-8 -*-

from pymongo import MongoClient
# from pyspark import SparkConf, SparkContext
import pandas as pd
# from operator import add
from collections import Counter
from relable import voting,relabel_cluster

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
# conf = (SparkConf()
    # .set("spark.driver.maxResultSize", "4g"))
# sc = SparkContext(conf=conf)
KN = 100
def co_association():
    DATA = []
    # data = [[{"1":[1,2,3]},{"2":[4,5]},{"3":[6,7]}],
            # [{"1":[6,7]},{"2":[1,2,3]},{"3":[4,5]}],
            # [{"1":[1,2]},{"2":[3,4]},{"3":[5,6,7]}]]
    # for i in range(1,5):
        # data = collection.find({'KN':i})[0]
        # temp = []
        # for cl in range(1,CLUSTERING+1):
            # temp.append({str(cl):data[str(cl)]})
        # DATA.append(temp)
    # data1 = []
    # for li in DATA:
        # for i in li:
            # for k,v in i.items():
                # data1.extend(sc.parallelize(v).map(lambda x:(x,v)).collect())
    # dataRDD = sc.parallelize(data1,3)
    # print "!@#$"*25
    # rdds = dataRDD.groupByKey().map(lambda x: (x[0],list(x[1]))).collect()
    # result = []
    # for i in rdds:
        # temp = []
        # for li in i[1]:
            # temp.extend(li)
        # result.append([i[0],Counter(temp)])
    # print result
    # print dataRDD.reduceByKey().collect() 
    clusters = []
    for K in range(KN):
        data = collection.find({'KN':K+1})[0]
        temp = [0 for i in range(POINTS)]
        for i in range(CLUSTERING):
            for point in data[str(i+1)]:
                temp[int(point)-1] = i+1
        clusters.append(temp)
    relabeled_clusters = relabel_cluster(clusters)
    # print relabeled_clusters[0][1]
    relabel_dict = {}
    for li in relabeled_clusters:
        for point in range(1,POINTS+1):
            if relabel_dict.has_key(li[point-1]):
                relabel_dict[li[point-1]].append(point)
            else:
                relabel_dict[li[point-1]] = [point]
    count = 0
    for i in range(1,CLUSTERING+1):
        co_ = Counter(relabel_dict[i])
        sup = []
        for k,v in co_.items():
            if v == KN:
                sup.append(k)
        count += len(sup)
    print count
if __name__=="__main__":
    co_association()