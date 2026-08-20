import os

# Kafka
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "network-conn")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "predictions")
BENIGN_TOPIC = os.getenv("BENIGN_TOPIC", "benign")
ATTACK_TOPIC = os.getenv("ATTACK_TOPIC", "attack")

# Model / data
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "pipeline"))
HISTORICAL_DATA_PATH = os.getenv("HISTORICAL_DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "historical.csv"))
STREAMING_DATA_PATH = os.getenv("STREAMING_DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "streaming.csv"))
CHECKPOINT_LOCATION = os.getenv("CHECKPOINT_LOCATION", os.path.join(os.path.dirname(__file__), "..", "checkpoint"))

# ML
CLASSIFICATION_MODE = os.getenv("CLASSIFICATION_MODE", "binary")  # "binary" or "multiclass"
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.8"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2057"))
NUM_ATTACK_LABELS = int(os.getenv("NUM_ATTACK_LABELS", "10"))

# API
SOURCE_API_URL = os.getenv("SOURCE_API_URL", "http://localhost:8000")
SINK_API_URL = os.getenv("SINK_API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
FETCH_BATCH_SIZE = int(os.getenv("FETCH_BATCH_SIZE", "1000"))
FETCH_INTERVAL = float(os.getenv("FETCH_INTERVAL", "1.0"))
FETCH_TYPE = os.getenv("FETCH_TYPE", "streaming")  # "historical" or "streaming"
FETCH_LOOP = os.getenv("FETCH_LOOP", "true").lower() in ("1", "true", "yes")
PUBLISH_BATCH_SIZE = int(os.getenv("PUBLISH_BATCH_SIZE", "1000"))
PUBLISH_TIMEOUT = float(os.getenv("PUBLISH_TIMEOUT", "5.0"))
PUBLISH_TOPIC = os.getenv("PUBLISH_TOPIC", OUTPUT_TOPIC)

# Mock API
MOCK_API_HOST = os.getenv("MOCK_API_HOST", "0.0.0.0")
MOCK_API_PORT = int(os.getenv("MOCK_API_PORT", "8000"))
MOCK_HISTORICAL_RECORDS = int(os.getenv("MOCK_HISTORICAL_RECORDS", "10000"))
MOCK_STREAMING_RECORDS = int(os.getenv("MOCK_STREAMING_RECORDS", "100000"))

# Feature / label columns
CATEGORICAL_COLS = ["protocol", "conn_state", "history", "service"]
NUMERIC_COLS = [
    "duration",
    "orig_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_bytes",
    "resp_pkts",
    "resp_ip_bytes",
    "missed_bytes",
]
BOOLEAN_COLS = ["local_orig", "local_resp"]
STRING_COLS = ["src_ip", "dest_ip"]
PORT_COLS = ["src_port", "dest_port"]

LABEL_COL = "label_multi"
