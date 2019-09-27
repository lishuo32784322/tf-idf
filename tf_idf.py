#coding=utf-8
import os
import re
import jieba
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


def f(sentenceData,spark):
    sentenceData = sentenceData.rdd.map(lambda x:(x[0], ' '.join(jieba.cut(re.sub(pattern,'',x[1]+x[2]),cut_all=False))))
    sentenceData = spark.createDataFrame(sentenceData).toDF("label", "sentence")
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    wordsData = tokenizer.transform(sentenceData)
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)
    featurizedData
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)
    for features_label in rescaledData.select("features", "label").take(3):    
        print(features_label)
#    result = rescaledData.select("label", "features")
#    result = result.rdd.flatMap(lambda x: (str(x[0]),str(x[1][1:])))
#    result.saveAsTextFile("/user/bigdata/tf_idf")
    

def main():
    conf = SparkConf().setMaster("yarn-client").setAppName('tf-idf').set('spark.executorEnv.PYTHONHASHSEED','0')
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel(logLevel='ERROR')
    hive_context = HiveContext(sc)
    sentenceData = hive_context.sql('select post_id,title,description from prod_bd_mysql_syn.dim_post_info limit 100')
    f(sentenceData,spark)


main()
