from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PIPELINE_DIR = BASE_DIR / "data_pipeline"
RAW_DIR = BASE_DIR / "data" / "raw"

REQUIRED_RAW_FILES = [
    RAW_DIR / "customers.csv",
    RAW_DIR / "articles.csv",
    RAW_DIR / "transactions_train.csv",
]

# MODE = "production"
# MODE = "dev"
MODE = "test"

MODE_CONFIG = {
    "test": {
        "OUTPUT_DIR": BASE_DIR / "data" / "processed",
    },
    "dev": {
        "OUTPUT_DIR": BASE_DIR / "data" / "processed",
    },
    "production": {
        "OUTPUT_DIR": BASE_DIR / "data" / "processed",
    },
}

RUNTIME_MODE = os.getenv("DATA_PIPELINE_MODE", MODE).strip().lower()

if RUNTIME_MODE not in MODE_CONFIG:
    raise ValueError(f"Unsupported MODE: {RUNTIME_MODE}")

CONFIG = MODE_CONFIG[RUNTIME_MODE]
OUTPUT_DIR: Path = CONFIG["OUTPUT_DIR"]

PIPELINE_STEPS = [
    DATA_PIPELINE_DIR / "build_customer_features.py",
    DATA_PIPELINE_DIR / "build_article_features.py",
    DATA_PIPELINE_DIR / "build_article_price_map.py",
    DATA_PIPELINE_DIR / "build_item_master.py",
    DATA_PIPELINE_DIR / "build_customer_purchase_profile.py",
    DATA_PIPELINE_DIR / "build_user_persona_scores.py",
    DATA_PIPELINE_DIR / "build_item_persona_scores.py",
    DATA_PIPELINE_DIR / "build_sim_users.py",
    DATA_PIPELINE_DIR / "build_simulated_events.py",
    DATA_PIPELINE_DIR / "validate_simulated_events.py",
    DATA_PIPELINE_DIR / "build_event_splits.py",
    DATA_PIPELINE_DIR / "build_item_features.py",
    DATA_PIPELINE_DIR / "build_ranking_train_data.py",
    DATA_PIPELINE_DIR / "build_candidate_training_data.py",
]


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def log_stage(stage: str, start_time: float, **stats: object) -> None:
    elapsed = time.perf_counter() - start_time
    stats_text = " ".join(f"{key}={value}" for key, value in stats.items())
    message = f"stage={stage} elapsed_seconds={elapsed:.2f}"
    if stats_text:
        message = f"{message} {stats_text}"
    logging.info(message)


def resolve_required_file(file_path: Path, description: str) -> Path:
    if file_path.exists():
        return file_path
    raise FileNotFoundError(f"Missing {description}: {file_path}")


def validate_raw_files() -> None:
    for file_path in REQUIRED_RAW_FILES:
        resolve_required_file(file_path, "raw dataset file")


def build_step_environment() -> Dict[str, str]:
    env = dict(os.environ)
    env["DATA_PIPELINE_MODE"] = RUNTIME_MODE
    return env


def run_step(script_path: Path, env: Dict[str, str]) -> None:
    start_time = time.perf_counter()
    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        env=env,
        check=True,
    )
    log_stage("pipeline_step_complete", start_time, script=script_path.name)


def main() -> None:
    configure_logging()
    run_start = time.perf_counter()
    validate_raw_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(
        "mode=%s raw_dir=%s output_dir=%s",
        RUNTIME_MODE,
        RAW_DIR,
        OUTPUT_DIR,
    )

    env = build_step_environment()
    for script_path in PIPELINE_STEPS:
        run_step(script_path, env)

    log_stage(
        "data_pipeline_complete",
        run_start,
        mode=RUNTIME_MODE,
        steps=len(PIPELINE_STEPS),
    )


if __name__ == "__main__":
    main()
