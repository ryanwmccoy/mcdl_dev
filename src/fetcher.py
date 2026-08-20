import json
import logging
import time

from kafka import KafkaProducer

from api_client import fetch_records
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    INPUT_TOPIC,
    FETCH_BATCH_SIZE,
    FETCH_INTERVAL,
    FETCH_TYPE,
    FETCH_LOOP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )

    offset = 0
    while True:
        try:
            records = fetch_records(FETCH_TYPE, offset=offset, limit=FETCH_BATCH_SIZE)
        except Exception as e:
            logger.error("Failed to fetch records: %s", e)
            time.sleep(FETCH_INTERVAL)
            continue

        if not records:
            logger.info("No more records from API at offset %s", offset)
            if FETCH_LOOP:
                offset = 0
                time.sleep(FETCH_INTERVAL)
                continue
            break

        for record in records:
            key = record.get("src_ip", "")
            producer.send(INPUT_TOPIC, key=key, value=record)

        producer.flush()
        logger.info("Published %s records to Kafka topic %s", len(records), INPUT_TOPIC)
        offset += len(records)
        time.sleep(FETCH_INTERVAL)

    producer.close()
    logger.info("Fetcher exiting")


if __name__ == "__main__":
    main()
