import logging
import requests

from config import (
    SOURCE_API_URL,
    SINK_API_URL,
    API_KEY,
    FETCH_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


def _headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def fetch_records(record_type, offset=0, limit=FETCH_BATCH_SIZE):
    url = f"{SOURCE_API_URL}/records"
    params = {"type": record_type, "offset": offset, "limit": limit}
    logger.info("Fetching records from %s (type=%s, offset=%s, limit=%s)", url, record_type, offset, limit)
    r = requests.get(url, params=params, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_all_records(record_type):
    records = []
    offset = 0
    while True:
        batch = fetch_records(record_type, offset=offset, limit=FETCH_BATCH_SIZE)
        if not batch:
            break
        records.extend(batch)
        offset += len(batch)
        if len(batch) < FETCH_BATCH_SIZE:
            break
    logger.info("Fetched %s total %s records", len(records), record_type)
    return records


def publish_records(records):
    url = f"{SINK_API_URL}/records"
    logger.info("Publishing %s tagged records to %s", len(records), url)
    r = requests.post(url, json=records, headers=_headers(), timeout=30)
    r.raise_for_status()
    return r.json()
