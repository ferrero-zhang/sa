from pyspark import HiveContext
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark import *
import os
import tempfile
import numpy as np  
import pandas as pd 
import json
import ijson

table = pd.read_csv('D:/github/sa/test.csv')
print(table['na'])

# conf = SparkConf().setMaster("yarn-client").set("spark.executor.memory", "512m").setAppName("SparkSQL").set("spark.eventLog.enabled",False)
# conf = SparkConf().setMaster("local").set("spark.executor.memory", "512m").setAppName("SparkSQL").set("spark.eventLog.enabled",False)
# conf.set("spark.home","/usr/hdp/2.5.0.0-1245/spark")
# sc = SparkContext(conf=conf)

# sqlContext = HiveContext(sc)

# df = sqlContext.read.json("D:/github/sa/na.json")
# df.show()
# df.printSchema()
# a = df.select("vote")
# a.show()
# df.select(df['name'], df['age'] + 1).show()
# df.where(df['vote'] != df['origin_vote']).select("data_id").show()
# dp = df.toPandas()
# print(dp.ix[[20],['origin_vote']].values.tolist()[0][0])
# print(type(dp.ix[[20],['1118']].values.tolist()))
# dp = pd.read_json(open('D:/github/sa/na.json','r+'))
# dp['na'] = 0

# for i in range(1,27):
    # for j in range(1,27):
        # dp['na'] += dp[str(i)+str(j)]*(dp[str(i)+str(j)]-1)/2
# print(dp['na'])
# dp.ix[[20],['1118']] += 1
# print(type(dp['1118'].values))
# dp.ix[[100],['S_id']] = "test"
# print(dp.ix[[100],['S_id']])
# df.write.format("orc").saveAsTable("PeopleWriteORC")