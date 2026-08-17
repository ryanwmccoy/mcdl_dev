from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    QuantileDiscretizer,
    VectorAssembler,
)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col, when

from config import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    BOOLEAN_COLS,
    PORT_COLS,
    STRING_COLS,
    LABEL_COL,
    CLASSIFICATION_MODE,
    RANDOM_SEED,
)


def cast_columns(df):
    """Cast known numeric, boolean, and port columns to correct Spark types."""
    for c in NUMERIC_COLS + PORT_COLS:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast("double"))

    for c in BOOLEAN_COLS:
        if c in df.columns:
            df = df.withColumn(
                c,
                when(col(c) == "true", 1.0)
                .when(col(c) == "false", 0.0)
                .when(col(c).isNull(), 0.0)
                .otherwise(col(c).cast("double")),
            )
    return df


def build_pipeline(num_classes=2, max_bins=32):
    """Build a Spark ML Pipeline that encodes, buckets, and classifies conn records."""
    indexers = [
        StringIndexer(
            inputCol=col_name,
            outputCol=f"{col_name}_idx",
            handleInvalid="keep",
        )
        for col_name in CATEGORICAL_COLS
    ]

    bucketizers = [
        QuantileDiscretizer(
            numBuckets=max_bins,
            inputCol=col_name,
            outputCol=f"{col_name}_bin",
            handleInvalid="keep",
        )
        for col_name in NUMERIC_COLS + PORT_COLS
    ]

    feature_cols = (
        [f"{c}_idx" for c in CATEGORICAL_COLS]
        + [f"{c}_bin" for c in NUMERIC_COLS + PORT_COLS]
        + BOOLEAN_COLS
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep",
    )

    if CLASSIFICATION_MODE == "binary":
        label_col = "label_bin"
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=label_col,
            numTrees=20,
            maxDepth=10,
            seed=RANDOM_SEED,
        )
    else:
        label_col = f"{LABEL_COL}_idx"
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=label_col,
            numTrees=20,
            maxDepth=10,
            seed=RANDOM_SEED,
        )

    return Pipeline(stages=indexers + bucketizers + [assembler, rf]), feature_cols, label_col


def prepare_label(df):
    """Derive the label column based on the configured classification mode."""
    if CLASSIFICATION_MODE == "binary":
        return df.withColumn(
            "label_bin",
            when(col(LABEL_COL) == "none", 1.0).otherwise(0.0),
        )
    else:
        label_indexer = StringIndexer(
            inputCol=LABEL_COL,
            outputCol=f"{LABEL_COL}_idx",
            handleInvalid="keep",
        )
        return label_indexer.fit(df).transform(df).withColumn(
            f"{LABEL_COL}_idx", col(f"{LABEL_COL}_idx").cast("double")
        )
