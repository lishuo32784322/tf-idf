import os
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import HiveContext
from pyspark.sql.window import Window
from pyspark.sql import Row, functions as F
from pyspark.ml.feature import HashingTF,IDF,Tokenizer


os.environ["PYSPARK_PYTHON"] = "/usr/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"
pattern = '[^\u4e00-\u9fa5a-zA-Z0-9]|\n|"'

conf = SparkConf().setMaster("yarn-client").setAppName('tf-idf').set('spark.executorEnv.PYTHONHASHSEED','0')
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel(logLevel='ERROR')
df = sc.textFile('/user/bigdata/nlp/word2vec.txt')
#df = spark.read.csv('file:///mnt/bigdata_workspace/TF-IDF/word2vec.txt')
print(df,type(df))
print(df.count())
