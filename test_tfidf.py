# coding:utf-8
# author:ls
import time, datetime, json, re, os, sys, random, shutil
"""
 计算TF-IDF
 @Time    : 2019/2/18 18:03
 @Author  : MaCan (ma_cancan@163.com)
 @File    : text_transformator.py
"""

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import *
from pyspark import SparkConf, SparkContext
from pyspark.sql import HiveContext
from pyspark.sql import Row, functions as F
from pyspark.sql import Window
from pymemcache.client.base import Client
import jieba
import os
import re

#过滤的pattern
remove_pattern = '[^\u4e00-\u9fa5a-zA-Z0-9]|\n|"'


def seg(data):
    """
    分词后返回分词的dataframe
    :param spark:
    :param data:
    :return:
    """
#    return [w for w in jieba.cut(data.strip(), cut_all=False) if len(w) > 1 and re.match(remove_pattern, w) is not None]
    return [w for w in jieba.cut(re.sub(remove_pattern,' ',data),cut_all=False) if len(w)>1]


def calc_tf(line):
    """
    计算每个单词在每篇文章的tf
    :param line:
    :return:
    """
    cnt_map = {}
    for w in line[2]:
        cnt_map[w] = cnt_map.get(w, 0) + 1
    lens = len(line[2])
#    if cnt_map.items():
    return [(line[0], (w, cnt *1.0/lens)) for w,cnt in cnt_map.items()]

def flat_map_2(line):
    rst = []
    idf_value = line[1][-1] * 1.0 / brocast_count.value
    if line[1][:-1]:
        for doc_pair in line[1][:-1]:
            for p in doc_pair:
                rst.append(Row(docId=p[0], token=line[0], tf_value=p[1], idf_value=idf_value, tf_idf_value=p[1] * idf_value)) 
    return rst

def f(rdd):
    mc = Client(('10.0.1.7',11211))
    for i in rdd:
        try:
            key = 'bd_post_tfidf_'+str(i[0])
            dic = {}
            for j in i[1]:
                if i[1]:
                    dic[j[0]]=j[1]
            res = ','.join([str(i[0])+':'+str(i[1]) for i in sorted(dic.items(),key=lambda dic:dic[1],reverse=True)[:20]])
            print('e'*6,key,res)
            mc.set(key,bytes(res,encoding='utf-8'))
        except Exception as e:
            print('q'*6,e,i,'\t',rdd)
        

if __name__ == '__main__':
    conf = SparkConf().setMaster("yarn-client").setAppName('tf-idf').set('spark.executorEnv.PYTHONHASHSEED','0')
    spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel(logLevel='ERROR')
    hive_context = HiveContext(sc)
    data = hive_context.sql('select post_id,title,description from prod_bd_mysql_syn.dim_post_info where title !="" or description != "" limit 1000')
    count = data.count()
    brocast_count = sc.broadcast(count)
    data = data.rdd.map(lambda x: (x[0], x[1]+x[2], seg(x[1]+' '+x[2]))).filter(lambda x:x[2])  # 分词
    tf_data = data.flatMap(calc_tf)
    idf_rdd = spark.createDataFrame(tf_data.map(lambda x: (x[1][0], x[0], x[1][1])),StructType([StructField('word',StringType(), True),StructField('id',StringType(),True),StructField('tf',FloatType(),True)]))
    word_count = spark.createDataFrame(tf_data.map(lambda x: (x[1][0],1)).reduceByKey(lambda x,y:x+y),StructType([StructField('word',StringType(), True),StructField('count',LongType(),True)]))
    tf_idf_df = idf_rdd.join(word_count,['word'],'left').na.fill(1).selectExpr('id','word','tf*count as tf_idf')
    tf_idf_df.withColumn('word_val',F.concat_ws(':',tf_idf_df.word,tf_idf_df.tf_idf)).createOrReplaceTempView('table')
    tf_idf_df = hive_context.sql('select id,concat_ws(",",collect_list(word_val)) as result from table group by id')
    print(tf_idf_df.take(20))
    
    
    
    #idf_rdd = idf_rdd.map(flat_map_2)
    spark.stop()






