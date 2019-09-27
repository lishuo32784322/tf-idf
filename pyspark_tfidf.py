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

def flat_with_doc_id(data):
    """
    flat map的时候带上文章ID
    :param data:
    :return:
    """
#    if data[2]:
    return [(data[0], w) for w in data[2]]


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

def create_combiner(v):
    t = []
    t.append(v)
    if v[0] =='' or v[1]=='':
        print('*'*20,v)
    return (t, 1)

def merge(x, v):
    t = []
    if x[0] is not None:
        t = x[0]
    t.append(v)
#    print('#'*20,x,v,t)
    return (t, x[1] + 1)

def merge_combine(x, y):
    t1 = []
    t2 = []
    if x[0] is not None:
        t1 = x[0]
    if y[0] is not None:
        t2 = y[0]
    t1 = t1.extend(t2)
#    print('*#'*10,t1,t2,x,y)
    return (t1, x[1] + y[1])

def flat_map_2(line):
    rst = []
    try:
        idf_value = line[1][-1] * 1.0 / brocast_count.value
        if line[1][:-1]:
            for doc_pair in line[1][:-1]:
                for p in doc_pair:
                    rst.append(Row(docId=p[0], token=line[0], tf_value=p[1], idf_value=idf_value, tf_idf_value=p[1] * idf_value)) 
    except Exception as e:print('wwwwww',e,line)
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
    data = hive_context.sql('select post_id,title,description from prod_bd_mysql_syn.dim_post_info where title !="" or description != "" ')
    count = data.count()
    brocast_count = sc.broadcast(count)
    data = data.rdd.map(lambda x: (x[0], x[1]+x[2], seg(x[1]+' '+x[2]))).filter(lambda x:x[2])  # 分词
    tf_data = data.flatMap(calc_tf)
    idf_rdd = tf_data.map(lambda x: (x[1][0], (x[0], x[1][1])))
    print(idf_rdd.take(10),idf_rdd.count(),1)
    idf_rdd = idf_rdd.combineByKey(create_combiner, merge, merge_combine)
    print(idf_rdd.take(10),idf_rdd.count(),2)
    idf_rdd = idf_rdd.map(flat_map_2)
    print(idf_rdd.take(10),idf_rdd.count(),3)
    idf_rdd.foreachPartition(f)
    tf_idf.repartition(1).saveAsTextFile("/user/bigdata/nlp/tf_idf.txt")
    print(12312312312)
    time.sleep(60)
    spark.stop()






