"""
core/state.py — атомарная запись состояния обучения.
Windows-safe: используем retry + filelock вместо прямого os.replace.
История эпох хранится в SQLite (не теряется между запусками).
"""

import json
import os
import time
import threading
from core.config import STATE_FILE

_write_lock = threading.Lock()


def _r() -> dict:
    try:
        return json.loads(STATE_FILE.read_text("utf-8"))
    except Exception:
        return {}


def _w(d: dict):
    """Атомарная запись. Windows-safe retry при PermissionError."""
    with _write_lock:
        tmp = str(STATE_FILE) + ".tmp"
        for attempt in range(10):
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, default=str)
                # Windows иногда держит файл — retries
                for replace_try in range(5):
                    try:
                        os.replace(tmp, str(STATE_FILE))
                        return
                    except PermissionError:
                        time.sleep(0.05)
                # Если replace не прошёл — пишем напрямую
                with open(str(STATE_FILE), "w", encoding="utf-8") as f:
                    json.dump(d, f, ensure_ascii=False, default=str)
                return
            except Exception:
                time.sleep(0.1 * (attempt + 1))
        # Последняя попытка — прямая запись без tmp
        try:
            with open(str(STATE_FILE), "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, default=str)
        except Exception:
            pass


def _save_epoch_to_db(epoch_data: dict, version_id: int = None):
    """Сохраняем данные эпохи в SQLite для постоянной истории."""
    try:
        from core.db import get_conn

        conn = get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_epochs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch       INTEGER NOT NULL,
                train_loss  REAL,
                val_loss    REAL,
                train_ppl   REAL,
                val_ppl     REAL,
                lr          REAL,
                duration    REAL,
                is_best     INTEGER DEFAULT 0,
                ts          INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )"""
        )
        from core.db import save_epoch as _save_ep

        _save_ep(version_id, epoch_data)
    except Exception:
        pass


def get_all_epochs_from_db() -> list:
    """Возвращает все эпохи из БД."""
    try:
        from core.db import get_conn

        conn = get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_epochs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER, train_loss REAL, val_loss REAL,
                train_ppl REAL, val_ppl REAL, lr REAL, duration REAL,
                is_best INTEGER DEFAULT 0,
                ts INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )"""
        )
        rows = conn.execute(
            "SELECT * FROM training_epochs ORDER BY epoch ASC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def init(params: dict, vocab_size: int, n_tokens: int, n_batches: int):
    existing = _r()
    if existing.get("status") == "paused":
        existing["stop_requested"] = False
        _w(existing)
        return
    _w(
        {
            "status": "idle",
            "params": params,
            "vocab_size": vocab_size,
            "n_tokens": n_tokens,
            "n_train_batches": n_batches,
            "total_params": None,
            "current_epoch": 0,
            "current_batch": 0,
            "train_loss": None,
            "val_loss": None,
            "train_ppl": None,
            "val_ppl": None,
            "best_val_loss": None,
            "best_epoch": None,
            "acc": None,
            "lr": None,
            "history": [],
            "started_at": None,
            "elapsed_sec": 0,
            "eta_sec": None,
            "epoch_eta_sec": None,
            "total_epochs_planned": 999,
            "stop_requested": False,
            "pause_epoch": None,
            "error": None,
            "updated_at": time.time(),
        }
    )


def set_running():
    d = _r()
    if not d.get("started_at"):
        d["started_at"] = time.time()
    d.update({"status": "running", "stop_requested": False, "updated_at": time.time()})
    _w(d)


def set_total_params(n: int):
    d = _r()
    d["total_params"] = n
    _w(d)


def mark_epoch_start():
    d = _r()
    d["_epoch_start"] = time.time()
    _w(d)


def update_batch(epoch: int, batch: int, loss: float, acc: float):
    d = _r()
    now = time.time()
    started = d.get("started_at") or now
    elapsed = now - started
    n_batches = d.get("n_train_batches", 1)
    total_plan = d.get("total_epochs_planned", epoch + 5)
    done = (epoch - 1) * n_batches + batch
    total_b = total_plan * n_batches
    speed = done / max(elapsed, 1)
    eta = (total_b - done) / speed if speed > 0 else None
    ep_start = d.get("_epoch_start") or now
    ep_elapsed = now - ep_start
    ep_speed = batch / max(ep_elapsed, 1)
    ep_eta = (n_batches - batch) / ep_speed if ep_speed > 0 else None
    d.update(
        {
            "current_epoch": epoch,
            "current_batch": batch,
            "train_loss": round(loss, 5),
            "acc": round(acc * 100, 2),
            "elapsed_sec": round(elapsed),
            "eta_sec": round(eta) if eta else None,
            "epoch_eta_sec": round(ep_eta) if ep_eta else None,
            "updated_at": now,
        }
    )
    _w(d)


def update_epoch(
    epoch: int,
    tl: float,
    vl: float,
    vppl: float,
    tppl: float,
    lr: float,
    dur: float,
    version_id: int = None,
):
    d = _r()
    now = time.time()
    is_best = d.get("best_val_loss") is None or vl < d["best_val_loss"]
    if is_best:
        d["best_val_loss"] = round(vl, 5)
        d["best_epoch"] = epoch

    epoch_rec = {
        "epoch": epoch,
        "train_loss": round(tl, 5),
        "val_loss": round(vl, 5),
        "val_ppl": round(vppl, 2),
        "train_ppl": round(tppl, 2),
        "lr": lr,
        "duration": round(dur, 1),
        "is_best": is_best,
    }
    d.setdefault("history", []).append(epoch_rec)

    # Сохраняем в БД для постоянной истории
    _save_epoch_to_db(epoch_rec, version_id=version_id)

    d.update(
        {
            "current_epoch": epoch,
            "train_loss": round(tl, 5),
            "val_loss": round(vl, 5),
            "val_ppl": round(vppl, 2),
            "train_ppl": round(tppl, 2),
            "lr": lr,
            "_epoch_start": now,
            "updated_at": now,
        }
    )
    _w(d)


def request_stop():
    d = _r()
    d["stop_requested"] = True
    d["updated_at"] = time.time()
    _w(d)


def is_stop_requested() -> bool:
    return _r().get("stop_requested", False)


def save_pause(epoch: int, batch: int):
    d = _r()
    d.update(
        {
            "status": "paused",
            "pause_epoch": epoch,
            "pause_batch": batch,
            "updated_at": time.time(),
        }
    )
    _w(d)


def get_resume_point() -> tuple:
    d = _r()
    if d.get("status") == "paused":
        return d.get("pause_epoch", 0), d.get("pause_batch", 0)
    return 0, 0


def set_finished():
    d = _r()
    d["status"] = "finished"
    d["updated_at"] = time.time()
    _w(d)


def set_error(msg: str):
    d = _r()
    d.update({"status": "error", "error": msg, "updated_at": time.time()})
    _w(d)


def set_epochs_planned(n: int):
    d = _r()
    d["total_epochs_planned"] = n
    _w(d)


def read() -> dict:
    d = _r()
    # Подмешиваем полную историю из БД
    db_history = get_all_epochs_from_db()
    if db_history:
        d["history"] = db_history
    return d
