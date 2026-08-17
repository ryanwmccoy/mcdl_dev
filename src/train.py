import os
import argparse

from pyspark.sql import SparkSession

from config import (
    HISTORICAL_DATA_PATH,
    MODEL_PATH,
    TRAIN_FRACTION,
    RANDOM_SEED,
)
from preprocessing import cast_columns, build_pipeline, prepare_label


def main():
    parser = argparse.ArgumentParser(description="Train a PySpark ML pipeline on historical conn data")
    parser.add_argument("--data", default=HISTORICAL_DATA_PATH, help="Path to historical CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to save the trained pipeline")
    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("mcdl-train")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    df = spark.read.option("header", "true").option("inferSchema", "true").csv(args.data)
    df = cast_columns(df)
    df = prepare_label(df)

    train_df, _ = df.randomSplit([TRAIN_FRACTION, 1 - TRAIN_FRACTION], seed=RANDOM_SEED)

    num_classes = train_df.select("label_bin" if os.getenv("CLASSIFICATION_MODE", "binary") == "binary" else "label_multi_idx").distinct().count()
    pipeline, feature_cols, label_col = build_pipeline(num_classes=num_classes)

    print(f"Training {os.getenv('CLASSIFICATION_MODE', 'binary')} model ...")
    print(f"Feature columns: {feature_cols}")
    print(f"Label column: {label_col}")

    model = pipeline.fit(train_df)

    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    model.write().overwrite().save(os.path.abspath(args.model))
    print(f"Pipeline saved to {os.path.abspath(args.model)}")

    spark.stop()


if __name__ == "__main__":
    main()
