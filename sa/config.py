# -*- coding: utf-8 -*-

MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
MONGODB_DB = 'Ifeng'
CLUSTERING_NAME = "re1"
MONGODB_COLLECTION = CLUSTERING_NAME+'_kmeans'
MONGODB_POINT = CLUSTERING_NAME+'_voting_super'
MONGODB_NA = CLUSTERING_NAME+'_na'
KN = 80 #基础聚类个数
CLUSTERING = 25  #标签数
POINTS = 1657   #数据个数
RLABLE = [0 for i in range(POINTS)]
