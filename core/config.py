import json
import re
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
STORAGE_DIR = ROOT_DIR / "storage"
MODELS_DIR = STORAGE_DIR / "models"
CKPT_DIR = STORAGE_DIR / "checkpoints"
LOG_DIR = STORAGE_DIR / "logs"
DB_FILE = STORAGE_DIR / "textreator.db"
STATE_FILE = STORAGE_DIR / "train_state.json"
CONFIG_FILE = STORAGE_DIR / "config.json"
DATASETS_DIR = STORAGE_DIR / "datasets"  # пользовательские датасеты

for _d in [STORAGE_DIR, MODELS_DIR, CKPT_DIR, LOG_DIR, DATASETS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or "model"


def model_dir(slug: str) -> Path:
    path = MODELS_DIR / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_ckpt_dir(slug: str) -> Path:
    path = model_dir(slug) / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_logs_dir(slug: str) -> Path:
    path = model_dir(slug) / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_samples_dir(slug: str) -> Path:
    path = model_dir(slug) / "samples"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_vocab_file(slug: str) -> Path:
    return model_dir(slug) / "vocab.json"


def model_tokens_file(slug: str) -> Path:
    return model_dir(slug) / "tokens.npy"


def model_state_file(slug: str) -> Path:
    return model_dir(slug) / "train_state.json"


ADMIN_USERNAME = "savely"
ADMIN_PASSWORD = "180386q1"

PROFILES = {
    "nano": {
        "name": "Nano",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 512,
        "dropout": 0.1,
        "seq_len": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "desc": "~4M параметров. Быстро обучается, качество базовое. Рекомендуется для теста.",
    },
    "small": {
        "name": "Small",
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "d_ff": 1024,
        "dropout": 0.15,
        "seq_len": 128,
        "batch_size": 32,
        "lr": 5e-4,
        "desc": "~15M параметров. Хороший баланс скорости и качества.",
    },
    "medium": {
        "name": "Medium",
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "d_ff": 2048,
        "dropout": 0.2,
        "seq_len": 256,
        "batch_size": 16,
        "lr": 3e-4,
        "desc": "~50M параметров. Высокое качество, медленно на CPU.",
    },
    "custom": {
        "name": "Custom",
        "d_model": 256,
        "n_heads": 8,
        "n_layers": 4,
        "d_ff": 1024,
        "dropout": 0.15,
        "seq_len": 128,
        "batch_size": 32,
        "lr": 5e-4,
        "desc": "Ручная настройка всех параметров.",
    },
}

DEFAULT_CONFIG = {
    "text_file": str(ROOT_DIR / "text.txt"),
    "profile": "small",
    "server_port": 5000,
    "custom": dict(PROFILES["small"]),
}


def load() -> dict:
    if CONFIG_FILE.exists():
        data = json.loads(CONFIG_FILE.read_text("utf-8"))
        for k, v in DEFAULT_CONFIG.items():
            data.setdefault(k, v)
        return data
    return dict(DEFAULT_CONFIG)


def save(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


def active_params(cfg: dict) -> dict:
    key = cfg.get("profile", "small")
    if key == "custom":
        return dict(cfg.get("custom", PROFILES["small"]))
    return dict(PROFILES.get(key, PROFILES["small"]))
