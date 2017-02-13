# -*- coding: utf-8 -*-

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
MONGODB_DB = 'Ifeng'
CLUSTERING_NAME = "k1b"
MONGODB_COLLECTION = CLUSTERING_NAME+'_kmeans'
MONGODB_POINT = CLUSTERING_NAME+'_voting_super'
MONGODB_NA = CLUSTERING_NAME+'_na'
KN = 50 #基础聚类个数
CLUSTERING = 6  #标签数
POINTS = 2340   #数据个数
RLABLE = [0 for i in range(POINTS)]
