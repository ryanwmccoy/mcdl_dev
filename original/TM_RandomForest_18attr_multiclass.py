# -*- coding: utf-8 -*-
"""
Created on Wed May 18 02:56:50 2022

@author: Tom
"""

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as func
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
from generate_individual_labels_dataframe import *

#############
# Run variables
#############

countRuns = True

binaryClassFlag = False

runGBT = False

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
    
########################
# functions
########################
    
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
    df_renamed = rename_columns(source_df)
except Exception as e:
    printToLog("Error:  " + str(e), log_location)
    print(traceback.format_exc())    

if (binaryClassFlag):
    df_dict = create_dict_of_dfs(df_renamed, benign_label, label_col_name, percent_attack_data, num_top_labels, True, log_location)
else:
    df_dict = { "All label techniques" : df_renamed }
    printToLog("df prepared for multiclass problem", log_location)

printToLog("", log_location) 

for key in df_dict:
    printToLog("Starting df " + key, log_location)
    printToLog("Instance start-time to match csv record: " + localNow, log_location)
    #Create df of interest (a certain label technique)
    unbinned_df = df_dict[key]
    
    if unbinned_df.count() < not_enough_rows_threshhold:
        printToLog("df " + key + " contains only " + str(unbinned_df.count()) +
                   " items, analysis discarded by row threshold\n\n", log_location)
    else:
        printToLog(str(unbinned_df.count()) + " rows", log_location)
        printToLog("Binning start", log_location)

        #Esteban and Nithisha: performance metric code 1
        #put this right before the first possible call to df_binning.py functions
        begin_binning = datetime.datetime.now()
        #end performance metric code 1
        
        if binaryClassFlag:
            unbinned_df = unbinned_df.withColumn("label_bin", when((col("label_multi") != "none"), 0.0).otherwise(1.0))
        else:
            unbinned_df = genNominalBinnedDF(unbinned_df, "label_multi", 1.0, False) #creates new col label_multi_bin
            unbinned_df = unbinned_df.withColumn("label_bin", unbinned_df["label_multi_bin"].cast('double') )
            
        #Carry out the binning on the df
        numeric_percent_trim = 0.02
        nominal_percent_agg = 0.98
        replace_bool = False

        attrList_ip_addr = ['dest_ip','src_ip']
        attrList_port = ['dest_port', 'src_port']
        attrList_bool = ['local_orig', 'local_resp']
        attrList_nominal = ['protocol', 'conn_state', 'history','service']
        attrList_numeric = ['duration','orig_bytes','orig_pkts','orig_ip_bytes','resp_bytes','resp_pkts','resp_ip_bytes','missed_bytes']
        
        # %%
        conn_df = genFullBinnedDF(unbinned_df, attrList_ip_addr, attrList_port, attrList_bool, attrList_nominal, nominal_percent_agg, attrList_numeric, numeric_percent_trim, replace_bool).persist()
        printToLog("Binning finished", log_location)
        #Esteban and Nithisha: performance metric code 2
        #put this right after the last possible call to df_binning.py functions
        end_binning = datetime.datetime.now()
        #end performance metric code 2
        
        ########################
        # Build Feature cols
        ########################     
        
        feature_cols = []
            
        #called from dtypes in case I care about column type when building 
        #feature_cols in the future
        for clm, types in conn_df.dtypes:
            if "_bin" in clm:
                if not "label" in clm:
                    feature_cols.append(clm)
        
        # build feature vector for Classifiers
        assembler = VectorAssembler(inputCols = feature_cols, outputCol = "features")
        conn_df = assembler.transform(conn_df)
        
        printToLog("feature_cols - " + str(len(feature_cols)) + " - " + str(feature_cols), log_location)
        
        ########################
        # Model
        ########################   
        
        label_col_name = "label"
        
        train, test = conn_df.randomSplit([0.7, 0.3], seed = 2057)
        rf = RandomForestClassifier(featuresCol = "features", 
                                    labelCol = "label_bin")
                                                                   
                                                           
        for clm in feature_cols:
            printToLog(str(clm) + " distinct values: " + str(train.select(clm)
                                                                  .distinct()
                                                                  .count()), log_location)
        
        train.printSchema()
        
        #Esteban and Nithisha: performance metric code 3
        #put this right before calling <model>.fit(train)
        #LONG WINDED COMMENT
        #I specified which model in the variable name because I'm still running both models
        #from the same script (something I plan to fix eventually)
        #you can get away with a simpler variable name like begin_training
        begin_randForestTraining = datetime.datetime.now()
        #end performance metric code 3
        rfModel = rf.fit(train)
        
        #Esteban and Nithisha: performance metric code 4
        #put this right after calling <model>.fit(train). see LONG WINDED COMMENT
        end_randForestTraining = datetime.datetime.now()
        #end performance metric code 4
        
        printToLog("randomForest model fit", log_location)
                
        #############
        # Predictions
        #############   
        
        #Esteban and Nithisha: performance metric code 5
        #put this right before calling <model>.fit(test) see LONG WINDED COMMENT
        begin_randForestPredictions = datetime.datetime.now()
        #end performance metric code 5
        
        predictions = rfModel.transform(test)
        
        #Esteban and Nithisha: performance metric code 6
        #put this right after calling <model>.fit(test) see LONG WINDED COMMENT
        end_randForestPredictions = datetime.datetime.now()
        #end performance metric code 6
        
        predictions_and_labels = predictions.select(["prediction", 
                                                     label_col_name+"_bin"])
        predictions_and_labels.selectExpr("cast(prediction as int) prediction")
        metrics = MulticlassMetrics(predictions_and_labels.rdd.map(tuple))
         
        
        evaluator = MulticlassClassificationEvaluator(labelCol = label_col_name+"_bin", 
                                              predictionCol = "prediction")
        
        if binaryClassFlag:                   
            binary_metrics = BinaryClassificationMetrics(predictions_and_labels.select("prediction", "label_bin").rdd.map(tuple))
            areaUnderCurve = binary_metrics.areaUnderROC
        else:
            areaUnderCurve = "na"

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
        
        #Esteban and Nithisha: performance metric code 7
        #put this anywhere after all other performance metric code sections have run
        bin_time = (begin_binning - end_binning).total_seconds()
        train_time = (begin_randForestTraining - end_randForestTraining).total_seconds()
        test_time = (begin_randForestPredictions - end_randForestPredictions).total_seconds()
        #These commented lines can be added to your reporting statements
        #csv_append_string = csv_append_string + ("\nbin_time:" + "\t" + str(bin_time))
        #csv_append_string = csv_append_string + ("\ntrain_time:" + "\t" + str(train_time))
        #csv_append_string = csv_append_string + ("\ntest_time:" + "\t" + str(test_time))
        #end performance metric code 7


        printToLog("randomForest metrics finished", log_location)

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
                               falsePositive,
                               bin_time,
                               train_time,
                               test_time)
                               
        
            
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
                           "falsePositive",
                           "bin_time",
                           "train_time",
                           "test_time")
            with open(rf_results_location, 'a') as fd:
                if getsize(rf_results_location) == 0:
                    fd.write(header)
                    
                fd.write(buff)
                fd.close()
                
            

        #############
        # Gradient boosting! 
        # mostly copied from above
        #############

        if runGBT:
            gbt = GBTClassifier(featuresCol = "features",
                        labelCol = "label_bin")
                        
            begin_gbTraining = datetime.datetime.now()
            gbtModel = gbt.fit(train)
            end_gbTraining = datetime.datetime.now()

            printToLog("GBTClassifier model fit", log_location)
            
            begin_gbPredictions = datetime.datetime.now()
            gbPredictions = gbtModel.transform(test)
            end_gbPredictions = datetime.datetime.now()

            gbPredictions_and_labels = gbPredictions.select(["prediction", 
                                                         label_col_name+"_bin"])
            gbPredictions_and_labels.selectExpr("cast(prediction as int) prediction")
            gbMetrics = MulticlassMetrics(gbPredictions_and_labels.rdd.map(tuple))
            gbEval = MulticlassClassificationEvaluator(labelCol = label_col_name+"_bin", 
                                                       predictionCol = "prediction")
            if binaryClassFlag:                 
                gb_binary_metrics = BinaryClassificationMetrics(gbPredictions_and_labels.select("prediction", "label_bin").rdd.map(tuple))
                gb_areaUnderCurve = gb_binary_metrics.areaUnderROC
            else:
                gb_areaUnderCurve = "na"

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

            gb_train_time = (begin_gbTraining - end_gbTraining).total_seconds()
            gb_test_time = (begin_gbPredictions - end_gbPredictions).total_seconds()

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
                               gb_falsePositive
                               bin_time,
                               gb_train_time,
                               gb_test_time)
            
            with open(gb_results_location, 'a') as fd:
                if getsize(gb_results_location) == 0:
                    fd.write(header)
                    
                fd.write(gbuff)
                fd.close()
            
            printToLog("GBTClassifier metrics finished", log_location)

        print("\n\nEnd of for each loop " + key + "\n\n\n\n\n\n\n")
        printToLog("End of for each loop " + key + "\n", log_location)
        conn_df.unpersist()
        
printToLog("End run\n-----------\n----------\n\n", log_location)