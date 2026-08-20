import os
import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, struct, to_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    BooleanType,
)
from pyspark.ml import PipelineModel

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    BENIGN_TOPIC,
    ATTACK_TOPIC,
    MODEL_PATH,
    CHECKPOINT_LOCATION,
)
from preprocessing import cast_columns


def _build_conn_schema():
    return StructType(
        [
            StructField("src_ip", StringType(), True),
            StructField("dest_ip", StringType(), True),
            StructField("src_port", DoubleType(), True),
            StructField("dest_port", DoubleType(), True),
            StructField("protocol", StringType(), True),
            StructField("conn_state", StringType(), True),
            StructField("service", StringType(), True),
            StructField("history", StringType(), True),
            StructField("duration", DoubleType(), True),
            StructField("orig_bytes", DoubleType(), True),
            StructField("orig_pkts", DoubleType(), True),
            StructField("orig_ip_bytes", DoubleType(), True),
            StructField("resp_bytes", DoubleType(), True),
            StructField("resp_pkts", DoubleType(), True),
            StructField("resp_ip_bytes", DoubleType(), True),
            StructField("missed_bytes", DoubleType(), True),
            StructField("local_orig", BooleanType(), True),
            StructField("local_resp", BooleanType(), True),
            StructField("label_multi", StringType(), True),
        ]
    )


def _kafka_query(df, bootstrap_servers, topic, checkpoint_base, checkpoint_name, output_cols):
    value_df = df.withColumn("value", to_json(struct(*output_cols))).select("value")
    return (
        value_df.writeStream
        .outputMode("append")
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("topic", topic)
        .option("checkpointLocation", os.path.join(os.path.abspath(checkpoint_base), checkpoint_name))
        .queryName(f"kafka-{topic}")
        .start()
    )


def main():
    parser = argparse.ArgumentParser(description="Run streaming inference on conn records from Kafka")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to trained pipeline")
    parser.add_argument("--input-topic", default=INPUT_TOPIC, help="Kafka input topic")
    parser.add_argument("--output-topic", default=OUTPUT_TOPIC, help="Kafka output topic for all predictions")
    parser.add_argument("--benign-topic", default=BENIGN_TOPIC, help="Kafka topic for benign predictions")
    parser.add_argument("--attack-topic", default=ATTACK_TOPIC, help="Kafka topic for attack predictions")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--checkpoint", default=CHECKPOINT_LOCATION, help="Checkpoint directory")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("mcdl-stream-inference")
        .config("spark.sql.streaming.checkpointLocation", os.path.abspath(args.checkpoint))
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )
        .getOrCreate()
    )

    schema = _build_conn_schema()

    raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", args.bootstrap)
        .option("subscribe", args.input_topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed = raw.select(from_json(col("value").cast("string"), schema).alias("data")).select("data.*")
    parsed = cast_columns(parsed)

    model = PipelineModel.load(os.path.abspath(args.model))
    predictions = model.transform(parsed)

    output_cols = ["src_ip", "dest_ip", "src_port", "dest_port", "protocol", "label_multi", "prediction"]
    output_df = predictions.select(*output_cols)

    # Console sink
    query_console = (
        output_df.writeStream
        .outputMode("append")
        .format("console")
        .option("truncate", "false")
        .queryName("console-predictions")
        .start()
    )

    # Kafka sinks: all predictions, benign, and attack
    _kafka_query(output_df, args.bootstrap, args.output_topic, args.checkpoint, "predictions", output_cols)
    _kafka_query(
        output_df.filter(col("prediction") == 1.0),
        args.bootstrap,
        args.benign_topic,
        args.checkpoint,
        "benign",
        output_cols,
    )
    _kafka_query(
        output_df.filter(col("prediction") == 0.0),
        args.bootstrap,
        args.attack_topic,
        args.checkpoint,
        "attack",
        output_cols,
    )

    print("Streaming inference started. Listening on topic:", args.input_topic)
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
