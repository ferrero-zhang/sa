from pyspark import HiveContext
from pyspark import SQLContext
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark import *
import os
import tempfile
os.chdir(tempfile.mkdtemp())
# conf = SparkConf().setMaster("yarn-client").set("spark.executor.memory", "512m").setAppName("SparkSQL").set("spark.eventLog.enabled",False)
conf = SparkConf().setMaster("local").set("spark.executor.memory", "512m").setAppName("SparkSQL").set("spark.eventLog.enabled",False)
# conf.set("spark.home","/usr/hdp/2.5.0.0-1245/spark")
sc = SparkContext(conf=conf)

sqlContext = HiveContext(sc)

df = sqlContext.read.json("D:/github/sa/people.json")
df.show()
df.printSchema()
# a = df.select("name")
# a.show()
# df.select(df['name'], df['age'] + 1).show()
# df.filter(df['age'] > 21).show()
# df.write.format("orc").saveAsTable("PeopleWriteORC")