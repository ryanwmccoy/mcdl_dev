# -*- coding: utf-8 -*-
"""
Created on Wed May 18 02:56:50 2022

@author: Tom
"""

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as f
from pyspark.sql.types import IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import Bucketizer
from pyspark.ml.feature import StringIndexer,VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.evaluation import BinaryClassificationMetrics
import traceback
import sys
import datetime
from os.path import getsize

from df_binning import *
#############
# Run variables
#############

countRuns = True

benign_label = "none"
label_col_name = "label_multi"
percent_attack_data = 0.3
num_top_labels = 10
not_enough_rows_threshhold = 500

conn_server_loc = 'hdfs://hadoop-master:9000/bagui-class/RPlenkers/Datasets/2022-05-23/Full_Dataset'

rf_results_location = '/home/tmcelroy/results/randomForest/' + str(datetime.date.today()) + '_results.csv'
gb_results_location = '/home/tmcelroy/results/gradientBoost/' + str(datetime.date.today()) + '_results.csv'
log_location = '/home/tmcelroy/logs/randomForest/' + str(datetime.date.today()) + '_log.txt'

drop_cols = ["src_ip", \
             "dest_ip", \
             "uid", \
             "community_id", \
             "local_orig", \
             "ts", \
             "service", \
             "datetime", \
             "history", \
             "duration"]
    
#takes any number of arguments, all should be objects or lists (nested lists not supported)
#produces a string formatted as a single line of a .csv file
#   a single \n is pre-pended to the output string to start a new row in the csv appended to
#   if argument is an object, casts object with str() and produces value like "object",
#   if argument is a list like [1, 2, 3], casts each object on the list with str() and
#       produces value like "(1) (2) (3)",
#
#Sample Usage:
#    buff = csvAppendBuffer(key,                #string
#                           percent_attack_data,#float
#                           drop_cols,          #list of string
#                           feature_cols,       #list of string
#                           accuracy)           #float
#
#    save_results_location = '/home/tmcelroy/results.csv'
#
#    with open(save_results_location, 'a') as fd:
#       fd.write(buff)
def csvAppendBuffer(*addMe):
    result = '\n'
    for thing in addMe:
        result += '"'
        if (type(thing) == list):
            result += '('
            result += ") (".join(str(itm) for itm in thing)
            result += ')'
        else:
            result += str(thing)
        result += '"'
        result += ','
    #-1 slice to get rid of a trailing ,
    return result[:-1]

########################
# code start
########################

spark = ( 
    SparkSession
    .builder
    .master("spark://hadoop-master:7077")
    .appName("TM_RandoForest")
    .config("spark.driver.cores", '4')
    .config("spark.driver.memory", "12g")
    .config("spark.executor.cores",'20')
    .config("spark.executor.memory", "40g")
    .config("spark.sql.shuffle.partitions",'240')
    .config("spark.submit.deployMode", "client")
    .getOrCreate()
    )
    
sc = spark.sparkContext    

localNow = str(datetime.datetime.now().strftime("%H:%M:%S"))
printToLog("Begin run\n-----------\n----------\n\n", log_location)
printToLog("Dataset location - " + conn_server_loc, log_location)

source_df = spark.read.option("header", True).options(inferSchema=True).csv(conn_server_loc)
 
try:
    source_df = rename_columns(source_df)
except Exception as e:
    printToLog("Error:  " + str(e), log_location)
    print(traceback.format_exc())   

for clm in drop_cols:
    source_df = source_df.drop(source_df[clm])

dict_of_dfs = create_dict_of_dfs(source_df, benign_label, label_col_name, percent_attack_data, num_top_labels, True, log_location)

printToLog("", log_location) 

for key in dict_of_dfs:
    printToLog("Starting df " + key, log_location)
    printToLog("Instance start-time to match csv record: " + localNow, log_location)
    
    conn_df = dict_of_dfs[key]
    
    if conn_df.count() < not_enough_rows_threshhold:
        printToLog("df " + key + " contains only " + str(conn_df.count()) +
                   " items, analysis discarded by row threshold\n\n", log_location)
    else:
        printToLog(str(conn_df.count()) + " rows", log_location)
        conn_df = conn_df.na.drop(how="any")
        # group our columns into functional groups as they need to be processed
        # booleans are also converted to ints
        numeric_cols = [name for name, types in conn_df.dtypes if types == 'int' \
                                                               or types == 'double' \
                                                               or types == 'bigint']
        
        string_cols = [name for name, types in conn_df.dtypes if types == 'string']
        
        bool_cols = [name for name, types in conn_df.dtypes if types == 'boolean']
        
        for clm in bool_cols:
            conn_df = conn_df.withColumn(clm+"_processed", 
                                         when(conn_df[clm] == 'false', 0).otherwise(1))
        
        ##########################
        # Pre-processing Pipelines
        ##########################
        
        indexers = [
            StringIndexer(inputCol=column, 
                          outputCol=column+"_processed").fit(conn_df) \
                    for column in string_cols
                    ]
        
        numeric_bucketing = [
            Bucketizer(
                splits=genNumericEdges(conn_df, x),
                inputCol=x,
                outputCol=x+"_processed"
                        ) for x in numeric_cols
                             ]
        
        stages_ = indexers + numeric_bucketing
        
        conn_df = Pipeline(stages=stages_).fit(conn_df).transform(conn_df)
        
        ########################
        # Build Feature cols
        ########################
        
        feature_cols = []
        
        #called from dtypes in case I care about column type when building 
        #feature_cols in the future
        for clm, types in conn_df.dtypes:
            if "_processed" in clm:
                if not label_col_name in clm:
                    feature_cols.append(clm)
        
        # build feature vector for Classifiers
        assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")
        conn_df = assembler.transform(conn_df)
        
        printToLog("feature_cols - " + str(feature_cols), log_location)
        
        ########################
        # Model
        ########################   
        
        train, test = conn_df.randomSplit([0.7, 0.3], seed = 2057)
        rf = RandomForestClassifier(featuresCol = "features", 
                                    labelCol = label_col_name+"_processed")
        gbt = GBTClassifier(featuresCol = "features",
                            labelCol = label_col_name+"_processed")
                                                             
        for clm in feature_cols:
            printToLog(str(clm) + " distinct values: " + str(train.select(clm)
                                                                  .distinct()
                                                                  .count()), log_location)
        
        rfModel = rf.fit(train)
        gbtModel = gbt.fit(train)
                
        #############
        # Predictions
        #############   
        
        predictions = rfModel.transform(test) 
        gbPredictions = gbtModel.transform(test)

        predictions_and_labels = predictions.select(["prediction", 
                                                     label_col_name+"_processed"])
        predictions_and_labels.selectExpr("cast(prediction as int) prediction")
        
        gbPredictions_and_labels = gbPredictions.select(["prediction", 
                                                     label_col_name+"_processed"])
        gbPredictions_and_labels.selectExpr("cast(prediction as int) prediction")
        
        metrics = MulticlassMetrics(predictions_and_labels.rdd.map(tuple)) 
        gbMetrics = MulticlassMetrics(gbPredictions_and_labels.rdd.map(tuple))
        
        evaluator = MulticlassClassificationEvaluator(labelCol = label_col_name+"_processed", 
                                                      predictionCol = "prediction")
        gbEval = MulticlassClassificationEvaluator(labelCol = label_col_name+"_processed", 
                                                   predictionCol = "prediction")
        
        binary_metrics = BinaryClassificationMetrics(predictions_and_labels.select("prediction", label_col_name+"_processed").rdd.map(tuple))
        
        gb_binary_metrics = BinaryClassificationMetrics(gbPredictions_and_labels.select("prediction", label_col_name+"_processed").rdd.map(tuple))
        
        cfsn_temp = metrics.confusionMatrix()
        cfsn_mtrx = np.array2string(cfsn_temp.toArray()).replace('\n', '')
        accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
        precision = evaluator.evaluate(predictions, {evaluator.metricName: "precisionByLabel",
                                                     evaluator.metricLabel: 1.0})
        recall = evaluator.evaluate(predictions, {evaluator.metricName: "recallByLabel",
                                                  evaluator.metricLabel: 1.0})
        f_measure = evaluator.evaluate(predictions, {evaluator.metricName: "fMeasureByLabel",
                                                     evaluator.metricLabel: 1.0})
        truePositive = evaluator.evaluate(predictions, {evaluator.metricName: "truePositiveRateByLabel",
                                                        evaluator.metricLabel: 1.0})
        falsePositive = evaluator.evaluate(predictions, {evaluator.metricName: "falsePositiveRateByLabel",
                                                         evaluator.metricLabel: 1.0}) 
        areaUnderCurve = binary_metrics.areaUnderROC


        gb_cfsn_temp = gbMetrics.confusionMatrix()
        gb_cfsn_mtrx = np.array2string(gb_cfsn_temp.toArray()).replace('\n', '')
        gb_accuracy = gbEval.evaluate(gbPredictions, {gbEval.metricName: "accuracy"})
        gb_precision = gbEval.evaluate(gbPredictions, {gbEval.metricName: "precisionByLabel",
                                                     gbEval.metricLabel: 1.0})
        gb_recall = gbEval.evaluate(gbPredictions, {gbEval.metricName: "recallByLabel",
                                                  gbEval.metricLabel: 1.0})
        gb_f_measure = gbEval.evaluate(gbPredictions, {gbEval.metricName: "fMeasureByLabel",
                                                     gbEval.metricLabel: 1.0})
        gb_truePositive = gbEval.evaluate(gbPredictions, {gbEval.metricName: "truePositiveRateByLabel",
                                                        gbEval.metricLabel: 1.0})
        gb_falsePositive = gbEval.evaluate(gbPredictions, {gbEval.metricName: "falsePositiveRateByLabel",
                                                         gbEval.metricLabel: 1.0})
        gb_areaUnderCurve = gb_binary_metrics.areaUnderROC

        ################################
        # Write to save_results_location
        ################################
        
        buff = csvAppendBuffer(localNow,
                               conn_server_loc,
                               key,
                               percent_attack_data,
                               len(feature_cols),
                               feature_cols,
                               cfsn_mtrx,
                               accuracy,
                               precision,
                               recall,
                               f_measure,
                               areaUnderCurve,
                               truePositive,
                               falsePositive)
                               
        gbuff = csvAppendBuffer(localNow,
                               conn_server_loc,
                               key,
                               percent_attack_data,
                               len(feature_cols),
                               feature_cols,
                               gb_cfsn_mtrx,
                               gb_accuracy,
                               gb_precision,
                               gb_recall,
                               gb_f_measure,
                               gb_areaUnderCurve,
                               gb_truePositive,
                               gb_falsePositive)
            
        if countRuns:
            header = csvAppendBuffer("Start", #localNow
                           "dataset", # conn_server_loc
                           "key",
                           "pct attack", #percent_attack_data,
                           "num features",
                           "feature_cols",
                           "cfsn_mtrx",
                           "accuracy",
                           "precision",
                           "recall",
                           "f_measure",
                           "areaUnderCurve",
                           "truePositive",
                           "falsePositive")
            with open(rf_results_location, 'a') as fd:
                if getsize(rf_results_location) == 0:
                    fd.write(header)
                    
                fd.write(buff)
                fd.close()
                
            with open(gb_results_location, 'a') as fd:
                if getsize(gb_results_location) == 0:
                    fd.write(header)
                    
                fd.write(gbuff)
                fd.close()

        print("\n\nEnd of for each loop " + key + "\n\n\n\n\n\n\n")
        printToLog("End of for each loop " + key + "\n", log_location)
        
printToLog("End run\n-----------\n----------\n\n", log_location)