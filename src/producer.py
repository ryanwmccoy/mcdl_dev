import os
import json
import time
import argparse
import csv

from kafka import KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    INPUT_TOPIC,
    STREAMING_DATA_PATH,
    NUMERIC_COLS,
    BOOLEAN_COLS,
    PORT_COLS,
)


def _coerce_record(row):
    """Convert CSV string values to the JSON types expected by the consumer schema."""
    for key in NUMERIC_COLS + PORT_COLS:
        if key in row and row[key]:
            try:
                row[key] = float(row[key])
            except ValueError:
                row[key] = 0.0
        else:
            row[key] = 0.0

    for key in BOOLEAN_COLS:
        if key in row:
            val = str(row[key]).strip().lower()
            row[key] = val in ("true", "1", "yes")
        else:
            row[key] = False
    return row


def main():
    parser = argparse.ArgumentParser(description="Publish conn records to Kafka")
    parser.add_argument("--data", default=STREAMING_DATA_PATH, help="CSV file to stream")
    parser.add_argument("--topic", default=INPUT_TOPIC, help="Kafka topic")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP_SERVERS, help="Kafka bootstrap servers")
    parser.add_argument("--rate", type=float, default=1.0, help="Records per second")
    args = parser.parse_args()

    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}. Run synthetic_data.py first.")

    delay = 1.0 / args.rate if args.rate > 0 else 0

    with open(args.data, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = _coerce_record(row)
            key = row.get("src_ip", "")
            producer.send(args.topic, key=key, value=row)
            print(f"Sent record to {args.topic}: {row.get('label_multi', 'unknown')}")
            time.sleep(delay)

    producer.flush()
    producer.close()


if __name__ == "__main__":
    main()
