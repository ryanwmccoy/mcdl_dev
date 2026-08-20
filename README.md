# MCDL Dockerized Kafka + PySpark Streaming ML Pipeline

This repo turns the original batch PySpark network-intrusion-detection scripts into a Dockerized, API-driven Apache Kafka + PySpark Structured Streaming pipeline. Records are pulled from a source API, classified, routed to per-prediction Kafka topics, and the tagged results are pushed back to a sink API.

## Architecture

```
+-----------+      /records      +---------+     network-conn     +------------------+
| source    | <------------------ | fetcher | -------------------> |                  |
| API       |                     +---------+                    |   Apache Kafka   |
| (mock-api)|                                                  -- |                  |
+-----------+                                                  |  |                  |
       ^                                                       |  +------------------+
       |                                                       |           |
       |                                                       |           v
       |                  /records (batches)                   |   +------------------+
       |                   POST back                           |   | stream-processor |
       |                                                       |   | (PySpark ML)     |
       |                                                       |   +------------------+
       |                                                       |           |
       |                                                       |    predictions
       |                                                       |    benign
       |                                                       |    attack
       |                                                       |           |
       |                                                       |           v
       |                                                       |   +------------------+
       +-------------------------------------------------------+-- | publisher        |
                                                                   +------------------+
```

1. **mock-api** – FastAPI source/sink that serves synthetic Zeek-style `conn` records and accepts tagged results.
2. **train** – Pulls historical records from the API, trains a Spark ML PipelineModel, and saves it to a shared volume.
3. **fetcher** – Pulls records from the API in `FETCH_BATCH_SIZE` (default 1000) increments and publishes them to Kafka `network-conn`.
4. **stream-processor** – Reads `network-conn`, loads the trained `PipelineModel`, classifies each record, and writes results to Kafka topics:
   - `predictions` – every classified record
   - `benign` – records predicted benign (`prediction == 1.0`)
   - `attack` – records predicted attack (`prediction == 0.0`)
5. **publisher** – Consumes `predictions` and POSTs batches of `PUBLISH_BATCH_SIZE` (default 1000) back to the sink API.

## Quick start (Docker)

1. Build images and start all services:
   ```bash
   docker compose up -d
   ```

2. Watch logs:
   ```bash
   docker compose logs -f stream-processor
   docker compose logs -f publisher
   ```

3. Check the mock API received count:
   ```bash
   curl http://localhost:8000/received
   ```

4. Read from a Kafka topic (using the Kafka host port):
   ```bash
   docker exec -it mcdl_dev-kafka-1 kafka-console-consumer \
     --bootstrap-server localhost:9092 --topic predictions --from-beginning
   ```

5. Stop everything:
   ```bash
   docker compose down
   ```

## Configuration

All values can be set through environment variables. Defaults are in `src/config.py`.

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker list |
| `INPUT_TOPIC` | `network-conn` | Raw conn records topic |
| `OUTPUT_TOPIC` | `predictions` | All classifications |
| `BENIGN_TOPIC` | `benign` | Benign classifications |
| `ATTACK_TOPIC` | `attack` | Attack classifications |
| `MODEL_PATH` | `models/pipeline` | Saved Spark PipelineModel path |
| `CHECKPOINT_LOCATION` | `checkpoint` | Spark Streaming checkpoint directory |
| `SOURCE_API_URL` | `http://localhost:8000` | API to fetch records from |
| `SINK_API_URL` | `http://localhost:8000` | API to push tagged records to |
| `API_KEY` | `""` | Bearer token added to API requests |
| `FETCH_BATCH_SIZE` | `1000` | Records fetched per API call |
| `FETCH_INTERVAL` | `1.0` | Seconds between fetches |
| `FETCH_TYPE` | `streaming` | `historical` or `streaming` |
| `FETCH_LOOP` | `true` | Whether to restart from offset 0 when exhausted |
| `PUBLISH_BATCH_SIZE` | `1000` | Tagged records per POST |
| `PUBLISH_TIMEOUT` | `5.0` | Max seconds before flushing a partial batch |
| `PUBLISH_TOPIC` | `predictions` | Topic consumed by the publisher |
| `CLASSIFICATION_MODE` | `binary` | `binary` (benign vs attack) or `multiclass` |
| `MOCK_HISTORICAL_RECORDS` | `10000` | Records served as `historical` |
| `MOCK_STREAMING_RECORDS` | `100000` | Records served as `streaming` |

## Running locally (without Docker)

1. Install Python dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Start Kafka:
   ```bash
   docker compose up -d zookeeper kafka
   ```

3. Start the mock API:
   ```bash
   python src/mock_api.py
   ```

4. Train the model from the API:
   ```bash
   python src/train.py --api-type historical
   ```

5. Start streaming inference:
   ```bash
   python src/stream_inference.py --model models/pipeline
   ```

6. Start the fetcher:
   ```bash
   python src/fetcher.py
   ```

7. Start the publisher:
   ```bash
   python src/publisher.py
   ```

## Original scripts

The original batch PySpark scripts are preserved in `original/` for reference.

## Notes

- Synthetic data is used by default so the pipeline is runnable without HDFS or an external Zeek log source.
- To use a real source/sink API, set `SOURCE_API_URL`, `SINK_API_URL`, and `API_KEY` to point at your endpoints. The expected contract is:
  - `GET /records?type=<historical|streaming>&offset=<int>&limit=<int>` returns a JSON list of records.
  - `POST /records` accepts a JSON list of tagged records.
- Kafka topics are auto-created. Downstream consumers can subscribe to `predictions`, `benign`, or `attack`.
