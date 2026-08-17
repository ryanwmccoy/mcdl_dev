import csv
import random
import ipaddress
import os
import argparse

from config import (
    HISTORICAL_DATA_PATH,
    STREAMING_DATA_PATH,
    NUMERIC_COLS,
    BOOLEAN_COLS,
    PORT_COLS,
    LABEL_COL,
    NUM_ATTACK_LABELS,
    RANDOM_SEED,
)

random.seed(RANDOM_SEED)

PROTOCOLS = ["tcp", "udp", "icmp"]
CONN_STATES = ["S0", "S1", "SF", "REJ", "RSTO", "RSTR", "RSTOS0", "OTH", "SH"]
SERVICES = ["http", "dns", "ftp", "smtp", "ssh", "ssl", "dhcp", "-", "other"]
HISTORIES = ["S", "Dd", "F", "ShADadFf", "-", "^", "C", "c"]


def _random_ip():
    return str(ipaddress.IPv4Address(random.randint(0, 2**32 - 1)))


def _generate_record(label=None):
    is_attack = label is not None and label != "none"
    record = {
        "src_ip": _random_ip(),
        "dest_ip": _random_ip(),
        "src_port": random.randint(1, 65535) if random.random() > 0.1 else 0,
        "dest_port": random.randint(1, 65535),
        "protocol": random.choice(PROTOCOLS),
        "conn_state": random.choice(CONN_STATES),
        "service": random.choice(SERVICES),
        "history": random.choice(HISTORIES),
        "duration": round(random.expovariate(0.5 if not is_attack else 0.1), 6),
        "orig_bytes": random.randint(0, 10000) if not is_attack else random.randint(1000, 500000),
        "orig_pkts": random.randint(0, 100) if not is_attack else random.randint(10, 5000),
        "orig_ip_bytes": random.randint(0, 20000) if not is_attack else random.randint(2000, 1000000),
        "resp_bytes": random.randint(0, 10000) if not is_attack else random.randint(100, 200000),
        "resp_pkts": random.randint(0, 100) if not is_attack else random.randint(1, 3000),
        "resp_ip_bytes": random.randint(0, 20000) if not is_attack else random.randint(500, 500000),
        "missed_bytes": random.randint(0, 1000),
        "local_orig": random.random() < 0.3,
        "local_resp": random.random() < 0.3,
    }
    record[LABEL_COL] = label if label is not None else "none"
    return record


def generate_labels():
    attack_labels = [f"T{1000 + i}" for i in range(NUM_ATTACK_LABELS)]
    return ["none"] + attack_labels


def generate_dataset(num_records, output_path, label_distribution=None):
    labels = generate_labels()
    if label_distribution is None:
        label_distribution = {label: 1.0 / len(labels) for label in labels}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = (
        ["src_ip", "dest_ip", "src_port", "dest_port"]
        + ["protocol", "conn_state", "service", "history"]
        + NUMERIC_COLS
        + BOOLEAN_COLS
        + [LABEL_COL]
    )

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for _ in range(num_records):
            label = random.choices(
                list(label_distribution.keys()),
                weights=list(label_distribution.values()),
                k=1,
            )[0]
            writer.writerow(_generate_record(label))


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic network conn data")
    parser.add_argument("--historical", type=int, default=10000, help="Number of historical records")
    parser.add_argument("--streaming", type=int, default=1000, help="Number of streaming records")
    args = parser.parse_args()

    labels = generate_labels()
    label_distribution = {label: 1.0 / len(labels) for label in labels}
    # make benign more common in historical data
    label_distribution["none"] = 0.6
    for label in labels:
        if label != "none":
            label_distribution[label] = 0.4 / (len(labels) - 1)

    print(f"Generating {args.historical} historical records -> {HISTORICAL_DATA_PATH}")
    generate_dataset(args.historical, HISTORICAL_DATA_PATH, label_distribution)

    print(f"Generating {args.streaming} streaming records -> {STREAMING_DATA_PATH}")
    generate_dataset(args.streaming, STREAMING_DATA_PATH, label_distribution)


if __name__ == "__main__":
    main()
