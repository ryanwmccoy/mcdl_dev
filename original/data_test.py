# -*- coding: utf-8 -*-
"""
Created on Fri June 10 12:56:50 2022

@author: Tom

change conn_server_loc to test a different dataset
change log_location to store results in a different location

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as func

import traceback
import sys
import datetime

#############
# Run variables
#############

countRuns = True

benign_label = "none"
label_col_name = "label_multi"
percent_attack_data = 0.3
num_top_labels = 3

conn_server_loc = 'hdfs://hadoop-master:9000/bagui-class/RPlenkers/Datasets/2022-05-23/Full_Dataset'

log_location = '/home/tmcelroy/logs/dataTest/' + str(datetime.date.today()) + '_log.txt'

col_names_dict = {"label_tactic":     "label_multi",
                  "service":          "service",
                  "orig_ip_bytes":    "orig_ip_bytes",
                  "local_resp":       "local_resp",
                  "missed_bytes":     "missed_bytes",
                  "proto":            "protocol",
                  "duration":         "duration",
                  "conn_state":       "conn_state",
                  "dest_ip_zeek":     "dest_ip",
                  "orig_pkts":        "orig_pkts",
                  "community_id":     "community_id",
                  "resp_ip_bytes":    "resp_ip_bytes",
                  "dest_port_zeek":   "dest_port",
                  "orig_bytes":       "orig_bytes",
                  "local_orig":       "local_orig",
                  "datetime":         "datetime",
                  "history":          "history",
                  "resp_bytes":       "resp_bytes",
                  "uid":              "uid",
                  "src_port_zeek":    "src_port",
                  "ts":               "ts",
                  "src_ip_zeek":      "src_ip"}
    
#############
# Functions
#############

# Function that renames the columns provided in col_names_dict.
def rename_columns(df_in, cols_dict):
    for c in cols_dict:
        if c not in df_in.columns:
            raise Exception("Cannot rename column " + str(c) + " to " + str(cols_dict[c]) + \
                            ".\n\t" + str(c) + " is not name of column in dataframe.\n")
        df_in = df_in.withColumnRenamed(c, cols_dict[c])
    return df_in

# Function that calculates the number of benign records required to meet a defined attack
# data percentage.  Requires user defined attack data percentage, and the number of attack
# records in the dataframe.  
def benign_data_count(attack_count, attack_pct):
    if attack_pct <= 0.0 or attack_pct > 1.0:
        raise Exception("Invalid Percent for Attack Records 0.0 < Attack Percent <= 1.0")
    return(((1.0 - attack_pct) * attack_count) / attack_pct)

def printToLog(addMe, logLocation):
    with open(logLocation, 'a') as fd:
        fd.write("" + str(datetime.datetime.now()) + ": " + str(addMe) + "\n")
########################
# code start
########################

spark = ( 
    SparkSession
    .builder
    .master("spark://hadoop-master:7077")
    .appName("TM_data_test")
    .config("spark.driver.memory", "12g")
    .config("spark.submit.deployMode", "client")
    .getOrCreate()
    )
sc = spark.sparkContext    

localNow = str(datetime.datetime.now().strftime("%H:%M:%S"))
printToLog("\n\n------------------------\nBegin run\n------------------------\n\n", log_location)
printToLog("Dataset location - " + conn_server_loc, log_location)

source_df = spark.read.option("header", True).options(inferSchema=True).csv(conn_server_loc)

#source_df = source_df.na.drop(how="any")

# Try/Catch block renames columns and catches exception if the provided 
# original column name is not in the dataframe. Error message is printed 
# to stderr, so the errors can be redirected to log file if desired. 
try:
    source_df = rename_columns(source_df, col_names_dict)
except Exception as e:
    print("Error:  " + str(e), file=sys.stderr)
    print(traceback.format_exc())   
col(label_col_name)

# Generates a dataframe that contains the counts of labels
df_unique_counts = (
    source_df
    .groupBy(label_col_name)
    .agg(func.count(func.lit(1)).alias("Num Of Records"))
    .sort(col("Num Of Records").desc())
                    )

df_unique_counts.show()

df_unique_counts = df_unique_counts.where(df_unique_counts[label_col_name] != benign_label)
df_benign = source_df.filter(source_df[label_col_name] == benign_label)
unique_non_benign = df_unique_counts.head(num_top_labels)
# One line for loop using dictionary comprehension.  Dictionary key is attack id (ex: T1595),
# and value is filtered dataframe containing all rows with key (ex: T1595) as its label.  
dict_of_dfs = {row[0]: source_df.filter(source_df[label_col_name] == row[0]) for row in unique_non_benign}

# for loop to add user defined number of records to meet user defined attack data percentage.  
for df_label in dict_of_dfs:
    benign_count = benign_data_count(dict_of_dfs[df_label].count(), percent_attack_data)

    printToLog("" + str(df_label) + " found in dataset with " + \
               str(dict_of_dfs[df_label].count()) + " unique records", log_location)

    df_tmp = df_benign.sample(False, ((benign_count % df_benign.count()) / df_benign.count()), 1234)  
    # Notes:  in python // performs integer division.  If int(benign_count)//int(df_benign.count()) == 0,
    # then the for loop does not execute.  Thus, no full benign dataframe copies are made.
    for i in range(0, (int(benign_count)//int(df_benign.count()))):
        df_tmp = df_tmp.unionByName(df_benign) # Use unionByName to copy full dataframes to df_tmp
    # Use unionByName to copy benign data to the attack dataframes in the dictionary. 
    dict_of_dfs[df_label] = dict_of_dfs[df_label].unionByName(df_tmp)