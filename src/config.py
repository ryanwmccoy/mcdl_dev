import os

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = os.getenv("INPUT_TOPIC", "network-conn")
OUTPUT_TOPIC = os.getenv("OUTPUT_TOPIC", "predictions")
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "..", "models", "pipeline"))
HISTORICAL_DATA_PATH = os.getenv("HISTORICAL_DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "historical.csv"))
STREAMING_DATA_PATH = os.getenv("STREAMING_DATA_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "streaming.csv"))
CHECKPOINT_LOCATION = os.getenv("CHECKPOINT_LOCATION", os.path.join(os.path.dirname(__file__), "..", "checkpoint"))

CLASSIFICATION_MODE = os.getenv("CLASSIFICATION_MODE", "binary")  # "binary" or "multiclass"
TRAIN_FRACTION = float(os.getenv("TRAIN_FRACTION", "0.8"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "2057"))
NUM_ATTACK_LABELS = int(os.getenv("NUM_ATTACK_LABELS", "10"))

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
