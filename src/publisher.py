import json
import logging
import time

from kafka import KafkaConsumer

from api_client import publish_records
from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    PUBLISH_TOPIC,
    PUBLISH_BATCH_SIZE,
    PUBLISH_TIMEOUT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    consumer = KafkaConsumer(
        PUBLISH_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id="mcdl-publisher",
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
    )

    buffer = []
    last_flush = time.time()
    logger.info("Publisher listening on topic %s", PUBLISH_TOPIC)

    try:
        for msg in consumer:
            buffer.append(msg.value)
            now = time.time()
            if len(buffer) >= PUBLISH_BATCH_SIZE or (now - last_flush) > PUBLISH_TIMEOUT:
                try:
                    publish_records(buffer)
                    consumer.commit()
                    logger.info("Published batch of %s records to sink API", len(buffer))
                except Exception as e:
                    logger.error("Failed to publish batch: %s", e)
                    # do not commit; reprocess on next poll
                buffer.clear()
                last_flush = now
    except KeyboardInterrupt:
        logger.info("Stopping publisher")
    finally:
        if buffer:
            try:
                publish_records(buffer)
                consumer.commit()
                logger.info("Published final batch of %s records", len(buffer))
            except Exception as e:
                logger.error("Failed to publish final batch: %s", e)
        consumer.close()


if __name__ == "__main__":
    main()
