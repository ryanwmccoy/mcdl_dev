import argparse
import csv
import datetime
import json
import os
import urllib.parse
from html.parser import HTMLParser

import numpy as np
import requests
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer, StringIndexer, VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql import SparkSession

from config import HISTORICAL_DATA_PATH, MODEL_PATH, RANDOM_SEED
from preprocessing import cast_columns

DEFAULT_UFW_URL = "https://datasets.uwf.edu/data/UWF-ZeekData24/parquet/"

# 12-attribute feature set from the original 12-attr multiclass script.
DROP_COLS = [
    "src_ip",
    "dest_ip",
    "uid",
    "community_id",
    "local_orig",
    "ts",
    "service",
    "datetime",
    "history",
    "duration",
]

SELECTED_FEATURES = [
    "src_port",
    "dest_port",
    "protocol",
    "conn_state",
    "orig_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "missed_bytes",
    "local_resp",
]

LABEL_COL = "label_multi"
INDEXED_LABEL = f"{LABEL_COL}_idx"


class _LinkExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href":
                    self.links.append(value)


def _should_skip(link: str) -> bool:
    decoded = urllib.parse.unquote(link).strip("/")
    if not decoded or decoded == ".." or decoded.startswith("_") or decoded.startswith("."):
        return True
    return False


def _download_parquet_index(url: str, dest_dir: str):
    """Recursively download .parquet files from an Apache-style directory index."""
    os.makedirs(dest_dir, exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    parser = _LinkExtractor()
    parser.feed(resp.text)

    for link in parser.links:
        if link in ("../", "", "/", "./"):
            continue
        if _should_skip(link):
            continue

        full_url = urllib.parse.urljoin(url, link)

        if link.endswith(".parquet"):
            rel_path = urllib.parse.unquote(link).lstrip("/")
            local_path = os.path.join(dest_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                print(f"Downloading {full_url} ...")
                r = requests.get(full_url, timeout=120)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(r.content)
        elif link.endswith("/"):
            subdir = urllib.parse.unquote(link).rstrip("/")
            _download_parquet_index(full_url, os.path.join(dest_dir, subdir))


def _resolve_data_path(args):
    if args.data.startswith(("http://", "https://")):
        _download_parquet_index(args.data, args.download_dir)
        return args.download_dir
    return args.data


def _load_data(spark, data_path, label_col):
    if os.path.isdir(data_path) or data_path.endswith(".parquet"):
        df = spark.read.parquet(data_path)
    else:
        df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

    # Map Zeek-style column names to the canonical names used by this repo.
    rename_map = {
        "proto": "protocol",
        "src_port_zeek": "src_port",
        "dest_port_zeek": "dest_port",
        "src_ip_zeek": "src_ip",
        "dest_ip_zeek": "dest_ip",
    }
    if label_col and label_col in df.columns:
        rename_map[label_col] = LABEL_COL
    else:
        # Fallback to any known label column.
        for candidate in ("label_tactic", "label_technique", "label_binary", "label_cve"):
            if candidate in df.columns:
                rename_map[candidate] = LABEL_COL
                break

    for old, new in rename_map.items():
        if old in df.columns and old != new:
            df = df.withColumnRenamed(old, new)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found after rename. Available columns: {df.columns}"
        )

    for c in DROP_COLS:
        if c in df.columns:
            df = df.drop(c)

    keep = [c for c in SELECTED_FEATURES + [LABEL_COL] if c in df.columns]
    df = df.select(*keep)
    df = cast_columns(df)
    # Remove rows with nulls in any feature or label so the RandomForest does not
    # see NaN/Infinity values in assembled vectors.
    df = df.dropna(subset=SELECTED_FEATURES + [LABEL_COL])
    return df


def _build_pipeline(df):
    categorical_cols = [c for c in SELECTED_FEATURES if dict(df.dtypes)[c] == "string"]
    numeric_cols = [c for c in SELECTED_FEATURES if c not in categorical_cols]

    label_indexer = StringIndexer(
        inputCol=LABEL_COL,
        outputCol=INDEXED_LABEL,
        handleInvalid="keep",
    )

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in categorical_cols
    ]

    bucketizers = [
        QuantileDiscretizer(
            numBuckets=8,
            inputCol=c,
            outputCol=f"{c}_bin",
            handleInvalid="keep",
        )
        for c in numeric_cols
    ]

    feature_cols = [f"{c}_idx" for c in categorical_cols] + [f"{c}_bin" for c in numeric_cols]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",
    )

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol=INDEXED_LABEL,
        numTrees=10,
        maxDepth=10,
        seed=RANDOM_SEED,
    )

    return Pipeline(stages=[label_indexer] + indexers + bucketizers + [assembler, rf]), feature_cols


def _collect_metrics(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol=INDEXED_LABEL,
        predictionCol="prediction",
    )

    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    weighted_precision = evaluator.evaluate(
        predictions, {evaluator.metricName: "weightedPrecision"}
    )
    weighted_recall = evaluator.evaluate(
        predictions, {evaluator.metricName: "weightedRecall"}
    )
    weighted_f1 = evaluator.evaluate(
        predictions, {evaluator.metricName: "weightedFMeasure"}
    )
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    pred_and_labels = (
        predictions.select("prediction", INDEXED_LABEL)
        .rdd.map(lambda r: (float(r[0]), float(r[1])))
    )
    mllib_metrics = MulticlassMetrics(pred_and_labels)
    confusion = mllib_metrics.confusionMatrix().toArray()
    confusion_str = np.array2string(
        confusion,
        separator=";",
        formatter={"float_kind": lambda x: f"{x:.0f}"},
        max_line_width=99999,
    ).replace("\n", " ")

    labels = sorted(
        int(row[0]) for row in predictions.select(INDEXED_LABEL).distinct().collect()
    )

    return {
        "accuracy": accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "f1": f1,
        "confusion_matrix": confusion_str,
        "mllib_metrics": mllib_metrics,
        "labels": labels,
    }


def _per_class_metrics(model, mllib_metrics, labels):
    label_indexer_model = model.stages[0]
    label_names = label_indexer_model.labels
    per_class = {}
    for label in labels:
        name = label_names[label] if label < len(label_names) else str(label)
        per_class[name] = {
            "precision": mllib_metrics.precision(float(label)),
            "recall": mllib_metrics.recall(float(label)),
            "f1": mllib_metrics.fMeasure(float(label), beta=1.0),
            "true_positive_rate": mllib_metrics.truePositiveRate(float(label)),
            "false_positive_rate": mllib_metrics.falsePositiveRate(float(label)),
        }
    return per_class


def _write_results(results_dir, row):
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, "multiclass_12attr_results.csv")
    header = [
        "timestamp",
        "data_source",
        "model_path",
        "feature_cols",
        "accuracy",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
        "f1",
        "confusion_matrix",
        "per_class_metrics",
    ]
    file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0

    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Results appended to {results_file}")


def _write_report(results_dir, metrics, per_class, model_path, data_source, feature_cols):
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(results_dir, f"multiclass_12attr_report_{timestamp}.txt")

    with open(report_file, "w") as f:
        f.write("12-Attribute Multiclass Random Forest Training Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp:    {datetime.datetime.now().isoformat()}\n")
        f.write(f"Data source:  {data_source}\n")
        f.write(f"Model path:   {model_path}\n")
        f.write(f"Feature cols: {feature_cols}\n\n")
        f.write("Overall Metrics\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy:            {metrics['accuracy']:.6f}\n")
        f.write(f"Weighted Precision:  {metrics['weighted_precision']:.6f}\n")
        f.write(f"Weighted Recall:     {metrics['weighted_recall']:.6f}\n")
        f.write(f"Weighted F1:         {metrics['weighted_f1']:.6f}\n")
        f.write(f"F1:                  {metrics['f1']:.6f}\n\n")
        f.write("Confusion Matrix (rows = true, columns = predicted)\n")
        f.write("-" * 40 + "\n")
        f.write(metrics["confusion_matrix"] + "\n\n")
        f.write("Per-Class Metrics\n")
        f.write("-" * 40 + "\n")
        for label, vals in sorted(per_class.items()):
            f.write(
                f"{label:20s}  precision={vals['precision']:.4f}  "
                f"recall={vals['recall']:.4f}  f1={vals['f1']:.4f}  "
                f"tpr={vals['true_positive_rate']:.4f}  fpr={vals['false_positive_rate']:.4f}\n"
            )

    print(f"Report written to {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Train a 12-attribute multiclass Random Forest on Zeek conn data"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_UFW_URL,
        help="Path to local CSV/parquet or URL to parquet directory index",
    )
    parser.add_argument(
        "--download-dir",
        default=os.path.join(os.path.dirname(HISTORICAL_DATA_PATH), "ufw_parquet"),
        help="Local directory to cache downloaded parquet files",
    )
    parser.add_argument(
        "--label-col",
        default="label_tactic",
        help="Source label column to use (renamed to label_multi)",
    )
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(MODEL_PATH), "multiclass_12attr"),
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--results",
        default=os.path.join(os.path.dirname(__file__) or ".", "..", "results"),
        help="Directory to write results CSV and report",
    )
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName("mcdl-multiclass-12attr")
        .master("local[*]")
        .config("spark.driver.memory", "3g")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    data_path = _resolve_data_path(args)
    df = _load_data(spark, data_path, args.label_col)
    print(f"Loaded {df.count():,} records; using {len(SELECTED_FEATURES)} feature columns")

    pipeline, feature_cols = _build_pipeline(df)
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=RANDOM_SEED)
    print(f"Training split: {train_df.count():,} | test split: {test_df.count():,}")

    model = pipeline.fit(train_df)
    predictions = model.transform(test_df)

    metrics = _collect_metrics(predictions)
    per_class = _per_class_metrics(model, metrics["mllib_metrics"], metrics["labels"])

    os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
    model.write().overwrite().save(os.path.abspath(args.model))
    print(f"Model saved to {os.path.abspath(args.model)}")

    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "data_source": args.data,
        "model_path": os.path.abspath(args.model),
        "feature_cols": json.dumps(feature_cols),
        "accuracy": metrics["accuracy"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
        "per_class_metrics": json.dumps(per_class),
    }
    _write_results(args.results, row)
    _write_report(
        args.results, metrics, per_class, os.path.abspath(args.model), args.data, feature_cols
    )

    print("\nMulticlass 12-attribute Random Forest results:")
    print(f"  Accuracy:            {metrics['accuracy']:.4f}")
    print(f"  Weighted Precision:  {metrics['weighted_precision']:.4f}")
    print(f"  Weighted Recall:     {metrics['weighted_recall']:.4f}")
    print(f"  Weighted F1:         {metrics['weighted_f1']:.4f}")
    print(f"  F1:                  {metrics['f1']:.4f}")
    print(f"  Confusion matrix:    {metrics['confusion_matrix']}")
    print(f"  Per-class metrics:   {json.dumps(per_class, indent=2)}")

    spark.stop()


if __name__ == "__main__":
    main()
