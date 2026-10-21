"""
monitor_and_retrain.py

Monitoring + retraining workflow for the casting defect classifier.

WHAT THIS DOES
---------------
1. MONITOR   — logs every production prediction (score, confidence, predicted
               label, and optional human-reviewed true label) to a local
               SQLite store, then computes rolling health metrics:
                 - average confidence
                 - rate of low-confidence ("uncertain") predictions
                 - predicted class balance drift vs. the training baseline
                 - live accuracy against any human-labeled feedback
2. TRIGGER   — evaluates those metrics against configurable thresholds plus
               a time-based schedule and a "enough new labeled data" rule to
               decide whether retraining is warranted.
3. RETRAIN   — builds a "challenger" model from the current architecture,
               trains it on the original training set PLUS newly collected
               human-labeled feedback images, and evaluates it on a held-out
               test set.
4. PROMOTE   — champion/challenger comparison: the challenger only replaces
               the production model if it beats it by at least
               `min_accuracy_improvement`. The previous champion is archived
               (never deleted) so you can roll back.

HOW TO WIRE IT IN
------------------
- Call `log_prediction(...)` from api.py's /predict endpoint after each
  inference (see snippet at the bottom of this file).
- When a human reviewer confirms/corrects a prediction (e.g. via a QA
  dashboard), call `record_feedback(...)` with the true label and save the
  image under FEEDBACK_DIR/<true_label>/. This is what feeds retraining.
- Run this script on a schedule (cron / Airflow / a simple `--daemon` loop)
  to check triggers and retrain automatically.

This is intentionally self-contained (SQLite + local filesystem, no
external services) so it runs anywhere. In a larger production setup, swap
the SQLite logger for a real metrics store (e.g. Prometheus/Postgres) and
the `--daemon` loop for an orchestrator (Airflow, Prefect, Dagster).
"""

import os
import io
import json
import shutil
import sqlite3
import logging
import argparse
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

import numpy as np
import tensorflow as tf
from typeguard import config

# ---------------------------------------------------------------------------
# Paths (mirrors the layout used by build_feature.py / train_model_from_store.py)
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

MODEL_STORE_DIR = os.path.join(BASE_DIR, "model_store")
ARCHIVE_DIR = os.path.join(MODEL_STORE_DIR, "archive")
CANDIDATE_DIR = os.path.join(MODEL_STORE_DIR, "candidates")
MONITOR_DIR = os.path.join(BASE_DIR, "monitor_store")
FEEDBACK_DIR = os.path.join(BASE_DIR, "feedback_data")  # human-labeled images from review

TRAIN_DATA_DIR = os.path.join(BASE_DIR, "data", "casting_data", "casting_data", "train")
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "casting_data", "casting_data", "test")

PROD_MODEL_PATH = os.path.join(MODEL_STORE_DIR, "casting_defect_classifier.keras")
DB_PATH = os.path.join(MONITOR_DIR, "monitoring.db")
RETRAIN_LOG_PATH = os.path.join(MONITOR_DIR, "retraining_runs.jsonl")

for d in (MODEL_STORE_DIR, ARCHIVE_DIR, CANDIDATE_DIR, MONITOR_DIR, FEEDBACK_DIR):
    os.makedirs(d, exist_ok=True)

IMG_SIZE = (224, 224)  # must match the deployed model's input shape
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("monitor_and_retrain")


# ---------------------------------------------------------------------------
# Config: thresholds that decide when retraining fires
# ---------------------------------------------------------------------------
@dataclass
class RetrainConfig:
    rolling_window: int = 500              # how many recent predictions to evaluate
    min_avg_confidence: float = 60.0        # trigger if rolling avg confidence drops below this (%)
    max_low_confidence_rate: float = 0.25   # trigger if >25% of predictions have confidence < 50%
    class_balance_drift_pp: float = 15.0    # trigger if predicted "defective" rate shifts >15pp vs baseline
    min_new_feedback_samples: int = 200     # trigger if this many new human-labeled images collected
    scheduled_retrain_days: int = 14        # trigger if it's been this long since the last retrain
    min_accuracy_improvement: float = 0.0   # challenger must beat champion by at least this to be promoted
    baseline_defective_rate: float = 0.5    # expected class balance from the original training set


# ---------------------------------------------------------------------------
# 1. MONITORING — prediction logging + metric computation
# ---------------------------------------------------------------------------
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT,
            score REAL NOT NULL,
            confidence REAL NOT NULL,
            predicted_label TEXT NOT NULL,
            true_label TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            trigger_reasons TEXT NOT NULL,
            challenger_accuracy REAL,
            champion_accuracy REAL,
            promoted INTEGER NOT NULL
        )
    """)
    return conn


def log_prediction(score: float, predicted_label: str, filename: str = None) -> None:
    """
    Call this from api.py right after inference to record every prediction
    the model makes in production.
    """
    confidence = abs(score - 0.50) * 2 * 100
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp, filename, score, confidence, predicted_label) "
            "VALUES (?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), filename, score, confidence, predicted_label),
        )


def record_feedback(filename: str, true_label: str, image_bytes: bytes = None) -> None:
    """
    Call this when a human reviewer confirms/corrects a prediction.
    - Updates the matching row's true_label (best-effort match on filename).
    - Optionally saves the image under FEEDBACK_DIR/<true_label>/ so it can
      be folded into the next retraining run.
    """
    assert true_label in ("ok", "defective"), "true_label must be 'ok' or 'defective'"

    with _get_conn() as conn:
        conn.execute(
            "UPDATE predictions SET true_label = ? WHERE filename = ? "
            "AND id = (SELECT id FROM predictions WHERE filename = ? ORDER BY id DESC LIMIT 1)",
            (true_label, filename, filename),
        )

    if image_bytes is not None:
        label_dir = os.path.join(FEEDBACK_DIR, true_label)
        os.makedirs(label_dir, exist_ok=True)
        out_path = os.path.join(label_dir, filename)
        with open(out_path, "wb") as f:
            f.write(image_bytes)


def compute_rolling_metrics(config: RetrainConfig) -> dict:
    """Pulls the most recent N predictions and summarizes model health."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT score, confidence, predicted_label, true_label FROM predictions "
            "ORDER BY id DESC LIMIT ?",
            (config.rolling_window,),
        ).fetchall()

    if not rows:
        return {"n": 0}

    scores = np.array([r[0] for r in rows])
    confidences = np.array([r[1] for r in rows])
    predicted_labels = [r[2] for r in rows]
    true_labels = [r[3] for r in rows]

    defective_rate = float(np.mean([p == "defective" for p in predicted_labels]))
    low_confidence_rate = float(np.mean(confidences < 50.0))

    labeled_pairs = [(p, t) for p, t in zip(predicted_labels, true_labels) if t is not None]
    live_accuracy = (
        float(np.mean([p == t for p, t in labeled_pairs])) if labeled_pairs else None
    )

    return {
        "n": len(rows),
        "avg_confidence": float(confidences.mean()),
        "low_confidence_rate": low_confidence_rate,
        "predicted_defective_rate": defective_rate,
        "n_labeled_feedback": len(labeled_pairs),
        "live_accuracy": live_accuracy,
    }


def days_since_last_retrain() -> float:
    if not os.path.exists(RETRAIN_LOG_PATH):
        return float("inf")
    with open(RETRAIN_LOG_PATH) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    if not lines:
        return float("inf")
    last_ts = datetime.fromisoformat(lines[-1]["timestamp"])
    return (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400


def count_new_feedback_samples() -> int:
    total = 0
    for label in ("ok", "defective"):
        label_dir = os.path.join(FEEDBACK_DIR, label)
        if os.path.isdir(label_dir):
            total += len([f for f in os.listdir(label_dir) if not f.startswith(".")])
    return total


# ---------------------------------------------------------------------------
# 2. TRIGGERS — decide whether retraining should fire
# ---------------------------------------------------------------------------
def check_triggers(config: RetrainConfig) -> dict:
    metrics = compute_rolling_metrics(config)
    reasons = []

    if metrics.get("n", 0) == 0:
        return {"should_retrain": False, "reasons": ["no predictions logged yet"], "metrics": metrics}

    if metrics["avg_confidence"] < config.min_avg_confidence:
        reasons.append(
            f"avg confidence {metrics['avg_confidence']:.1f}% < threshold {config.min_avg_confidence}%"
        )

    if metrics["low_confidence_rate"] > config.max_low_confidence_rate:
        reasons.append(
            f"low-confidence rate {metrics['low_confidence_rate']:.1%} > threshold "
            f"{config.max_low_confidence_rate:.1%}"
        )

    drift_pp = abs(metrics["predicted_defective_rate"] - config.baseline_defective_rate) * 100
    if drift_pp > config.class_balance_drift_pp:
        reasons.append(
            f"predicted class balance drifted {drift_pp:.1f}pp from baseline "
            f"(threshold {config.class_balance_drift_pp}pp) — possible data drift"
        )

    if metrics["live_accuracy"] is not None and metrics["live_accuracy"] < 0.90:
        reasons.append(f"live accuracy on reviewed feedback dropped to {metrics['live_accuracy']:.1%}")

    n_feedback = count_new_feedback_samples()
    if n_feedback >= config.min_new_feedback_samples:
        reasons.append(f"{n_feedback} new human-labeled samples collected (threshold {config.min_new_feedback_samples})")

    days_elapsed = days_since_last_retrain()
    if days_elapsed >= config.scheduled_retrain_days:
        reasons.append(f"{days_elapsed:.1f} days since last retrain (schedule: every {config.scheduled_retrain_days} days)")

    return {"should_retrain": len(reasons) > 0, "reasons": reasons, "metrics": metrics}


# ---------------------------------------------------------------------------
# 3. RETRAIN — build + train a challenger model
# ---------------------------------------------------------------------------
def _build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(*IMG_SIZE, 3)),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _merge_feedback_into_training_dir() -> str:
    """
    Copies newly labeled feedback images into a temp training directory
    alongside the original training set, so the challenger trains on both.
    Returns the path to that merged directory.
    """
    merged_dir = os.path.join(MONITOR_DIR, "merged_train")
    if os.path.exists(merged_dir):
        shutil.rmtree(merged_dir)

    for label in ("ok", "defective"):
        src = os.path.join(TRAIN_DATA_DIR, label)
        dst = os.path.join(merged_dir, label)
        os.makedirs(dst, exist_ok=True)
        if os.path.isdir(src):
            for fname in os.listdir(src):
                shutil.copy(os.path.join(src, fname), os.path.join(dst, fname))

        feedback_src = os.path.join(FEEDBACK_DIR, label)
        if os.path.isdir(feedback_src):
            for fname in os.listdir(feedback_src):
                shutil.copy(os.path.join(feedback_src, fname), os.path.join(dst, f"fb_{fname}"))

    return merged_dir


def train_challenger() -> tf.keras.Model:
    log.info("Merging original training data with newly collected feedback...")
    merged_dir = _merge_feedback_into_training_dir()

    import os

    # Add this check before calling image_dataset_from_directory
    data_dir = "monitor_store/merged_train"
    if not os.path.exists(data_dir) or not any(os.scandir(data_dir)):
        print("[INFO] No training images found in merged directory. Skipping retraining cycle.")
        return None

    train_ds = tf.keras.utils.image_dataset_from_directory(
        merged_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.15, subset="training", seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        merged_dir, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
        validation_split=0.15, subset="validation", seed=42,
    )

    model = _build_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    log.info("Training challenger model...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stopping])
    return model


def evaluate_on_test_set(model: tf.keras.Model) -> float:
    if not os.path.isdir(TEST_DATA_DIR):
        log.warning("Held-out test directory not found at %s — cannot evaluate.", TEST_DATA_DIR)
        return None
    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DATA_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False,
    )
    _, accuracy = model.evaluate(test_ds, verbose=0)
    return float(accuracy)


# ---------------------------------------------------------------------------
# 4. PROMOTE — champion/challenger comparison + safe rollout
# ---------------------------------------------------------------------------
def promote_if_better(challenger: tf.keras.Model, config: RetrainConfig) -> dict:
    challenger_acc = evaluate_on_test_set(challenger)

    champion_acc = None
    if os.path.exists(PROD_MODEL_PATH):
        champion = tf.keras.models.load_model(PROD_MODEL_PATH)
        champion_acc = evaluate_on_test_set(champion)

    promoted = False
    if challenger_acc is not None and (
        champion_acc is None or challenger_acc >= champion_acc + config.min_accuracy_improvement
    ):
        promoted = True
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

        # Archive the current champion before overwriting it (never delete).
        if os.path.exists(PROD_MODEL_PATH):
            archive_path = os.path.join(ARCHIVE_DIR, f"casting_defect_classifier_{timestamp}.keras")
            shutil.copy(PROD_MODEL_PATH, archive_path)
            log.info("Archived previous champion to %s", archive_path)

        challenger.save(PROD_MODEL_PATH)
        log.info("Promoted challenger to production: %s", PROD_MODEL_PATH)
    else:
        candidate_path = os.path.join(CANDIDATE_DIR, f"candidate_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.keras")
        challenger.save(candidate_path)
        log.info("Challenger did not beat champion — saved for review at %s, champion unchanged.", candidate_path)

    return {"challenger_accuracy": challenger_acc, "champion_accuracy": champion_acc, "promoted": promoted}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_cycle(config: RetrainConfig, force: bool = False) -> dict:
    challenger = train_challenger()
    if challenger is None:
        print("[INFO] Skipping promotion phase due to missing challenger model.")
        return

    result = promote_if_better(challenger, config)

    trigger_result = check_triggers(config)
    log.info("Metrics: %s", trigger_result["metrics"])

    if not (trigger_result["should_retrain"] or force):
        log.info("No retraining trigger fired. All clear.")
        return {"retrained": False, **trigger_result}

    log.info("Retraining triggered. Reasons: %s", trigger_result["reasons"] or ["manual/forced run"])
    challenger = train_challenger()
    result = promote_if_better(challenger, config)

    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_reasons": trigger_result["reasons"] or ["forced"],
        **result,
    }
    with open(RETRAIN_LOG_PATH, "a") as f:
        f.write(json.dumps(run_record) + "\n")
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO retrain_runs (timestamp, trigger_reasons, challenger_accuracy, champion_accuracy, promoted) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_record["timestamp"], json.dumps(run_record["trigger_reasons"]),
             result["challenger_accuracy"], result["champion_accuracy"], int(result["promoted"])),
        )

    return {"retrained": True, **trigger_result, **result}


def main():
    parser = argparse.ArgumentParser(description="Monitor the casting defect classifier and retrain on trigger.")
    parser.add_argument("--check", action="store_true", help="Only report metrics/triggers, don't retrain.")
    parser.add_argument("--retrain", action="store_true", help="Force a retraining run regardless of triggers.")
    parser.add_argument("--daemon", action="store_true", help="Loop forever, checking on an interval.")
    parser.add_argument("--interval-hours", type=float, default=6.0, help="Interval for --daemon mode.")
    args = parser.parse_args()

    config = RetrainConfig()

    if args.daemon:
        log.info("Starting monitoring daemon (checking every %.1fh)...", args.interval_hours)
        while True:
            run_cycle(config, force=False)
            time.sleep(args.interval_hours * 3600)
    elif args.check:
        result = check_triggers(config)
        print(json.dumps(result, indent=2))
    else:
        run_cycle(config, force=args.retrain)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Integration snippet for api.py (add inside predict_defect(), after scoring):
#
#   from monitor_and_retrain import log_prediction
#   ...
#   log_prediction(score=score, predicted_label=result_label, filename=file.filename)
# ---------------------------------------------------------------------------