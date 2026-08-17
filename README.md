# MCDL Kafka Streaming ML Pipeline

This repo turns the original batch PySpark ML classification scripts into a streaming Apache Kafka pipeline for real-time network connection classification.

## Architecture

```
CSV historical data  ->  train.py  ->  saved Spark PipelineModel
CSV streaming data   ->  producer.py  ->  Kafka topic "network-conn"
Kafka "network-conn" ->  stream_inference.py  ->  Kafka topic "predictions" + console
```

## Quick start

1. Start Kafka (requires Docker):
   ```bash
   docker compose up -d
   ```

2. Create the data and train a model:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   python src/synthetic_data.py --historical 10000 --streaming 1000
   python src/train.py --data data/historical.csv --model models/pipeline
   ```

3. Start the streaming inference job:
   ```bash
   python src/stream_inference.py --model models/pipeline
   ```

4. In another terminal, publish streaming records:
   ```bash
   python src/producer.py --data data/streaming.csv --rate 2
   ```

5. Watch predictions in the `stream_inference.py` console and on the `predictions` Kafka topic.

## Configuration

Environment variables (all have defaults):

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker list |
| `INPUT_TOPIC` | `network-conn` | Topic for raw conn records |
| `OUTPUT_TOPIC` | `predictions` | Topic for classification results |
| `MODEL_PATH` | `models/pipeline` | Where the trained pipeline is saved/loaded |
| `HISTORICAL_DATA_PATH` | `data/historical.csv` | Training data |
| `STREAMING_DATA_PATH` | `data/streaming.csv` | Streaming records to publish |
| `CHECKPOINT_LOCATION` | `checkpoint` | Spark Streaming checkpoint directory |
| `CLASSIFICATION_MODE` | `binary` | `binary` (attack vs benign) or `multiclass` |

## Original scripts

The original batch PySpark scripts are preserved in `original/` for reference.

## Notes

- The first run uses synthetic data so the pipeline is runnable without HDFS.
- To use real Zeek `conn.log` data, replace the CSV paths and adjust `src/config.py` columns as needed.
- `train.py` fits the full Spark ML Pipeline (StringIndexer, QuantileDiscretizer, VectorAssembler, RandomForest) and saves it so `stream_inference.py` can load the same preprocessing without re-fitting.
