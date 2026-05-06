import sqlite3
import hashlib
import secrets
import time
import json
from pathlib import Path
from core.config import DB_FILE, ADMIN_USERNAME, ADMIN_PASSWORD, slugify


def get_conn():
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _ensure_col(conn, table, col, defn):
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if col not in [r[1] for r in rows]:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {defn}")


def _ensure_model_slug_column(conn):
    rows = conn.execute("PRAGMA table_info(models)").fetchall()
    names = [r[1] for r in rows]
    if "slug" not in names:
        conn.execute("ALTER TABLE models ADD COLUMN slug TEXT")
        existing = set()
        for row in conn.execute("SELECT id, name FROM models").fetchall():
            base = slugify(row["name"])
            slug = base
            suffix = 1
            while slug in existing:
                slug = f"{base}-{suffix}"
                suffix += 1
            existing.add(slug)
            conn.execute("UPDATE models SET slug=? WHERE id=?", (slug, row["id"]))
        conn.commit()


def init_db():
    conn = get_conn()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        username  TEXT    UNIQUE NOT NULL,
        pw_hash   TEXT    NOT NULL,
        salt      TEXT    NOT NULL,
        role      TEXT    NOT NULL DEFAULT 'user',
        admin_ip  TEXT,
        created   INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token    TEXT    PRIMARY KEY,
        user_id  INTEGER NOT NULL REFERENCES users(id),
        expires  INTEGER NOT NULL,
        ip       TEXT
    );
    CREATE TABLE IF NOT EXISTS chats (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id  INTEGER NOT NULL REFERENCES users(id),
        title    TEXT    NOT NULL DEFAULT 'Новый чат',
        model_version_id INTEGER REFERENCES model_versions(id),
        created  INTEGER NOT NULL DEFAULT (strftime('%s','now')),
        updated  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS messages (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id  INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
        role     TEXT    NOT NULL CHECK(role IN ('user','bot')),
        content  TEXT    NOT NULL,
        ts       INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS word_corrections (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL REFERENCES users(id),
        msg_id    INTEGER REFERENCES messages(id) ON DELETE SET NULL,
        wrong     TEXT    NOT NULL,
        correct   TEXT    NOT NULL,
        context   TEXT,
        ts        INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS user_datasets (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id   INTEGER NOT NULL REFERENCES users(id),
        filename  TEXT    NOT NULL,
        filepath  TEXT    NOT NULL,
        words     INTEGER DEFAULT 0,
        ts        INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS models (
        id        INTEGER PRIMARY KEY AUTOINCREMENT,
        name      TEXT    NOT NULL,
        slug      TEXT    UNIQUE NOT NULL,
        created   INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS model_versions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id     INTEGER NOT NULL REFERENCES models(id),
        version_num  INTEGER NOT NULL DEFAULT 1,
        params       TEXT,
        vocab_size   INTEGER,
        best_epoch   INTEGER,
        best_val_loss REAL,
        checkpoint_path TEXT,
        notes        TEXT,
        created      INTEGER NOT NULL DEFAULT (strftime('%s','now')),
        updated      INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS training_epochs (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id    INTEGER REFERENCES model_versions(id),
        epoch         INTEGER NOT NULL,
        train_loss    REAL,
        val_loss      REAL,
        train_ppl     REAL,
        val_ppl       REAL,
        lr            REAL,
        duration      REAL,
        is_best       INTEGER DEFAULT 0,
        validated     INTEGER DEFAULT 0,
        custom_name   TEXT    DEFAULT '',
        ckpt_deleted  INTEGER DEFAULT 0,
        ts            INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS model_samples (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        version_id    INTEGER NOT NULL REFERENCES model_versions(id),
        epoch         INTEGER NOT NULL,
        sample_num    INTEGER NOT NULL,
        params        TEXT,
        text          TEXT,
        saved_name    TEXT,
        created       INTEGER NOT NULL DEFAULT (strftime('%s','now'))
    );
    CREATE TABLE IF NOT EXISTS gen_queue (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL REFERENCES users(id),
        request_id TEXT    NOT NULL UNIQUE,
        status     TEXT    NOT NULL DEFAULT 'waiting',
        prompt     TEXT,
        params     TEXT,
        result     TEXT,
        error      TEXT,
        created    INTEGER NOT NULL DEFAULT (strftime('%s','now')),
        started    INTEGER,
        finished   INTEGER
    );
    CREATE INDEX IF NOT EXISTS idx_msgs_chat   ON messages(chat_id);
    CREATE INDEX IF NOT EXISTS idx_corr_wrong  ON word_corrections(wrong);
    CREATE INDEX IF NOT EXISTS idx_ds_user     ON user_datasets(user_id);
    CREATE INDEX IF NOT EXISTS idx_mv_model    ON model_versions(model_id);
    CREATE INDEX IF NOT EXISTS idx_ep_version  ON training_epochs(version_id);
    CREATE INDEX IF NOT EXISTS idx_queue_user  ON gen_queue(user_id, status);
    """)
    conn.commit()
    try:
        _ensure_model_slug_column(conn)
    except Exception:
        pass
    # Migration: add admin_ip if missing
    try:
        _ensure_col(conn, "users", "admin_ip", "TEXT")
        _ensure_col(conn, "sessions", "ip", "TEXT")
        conn.commit()
    except Exception:
        pass

    # Create admin if not exists
    row = conn.execute(
        "SELECT id FROM users WHERE username=?", (ADMIN_USERNAME,)
    ).fetchone()
    if not row:
        salt = secrets.token_hex(16)
        pw_hash = hashlib.sha256((salt + ADMIN_PASSWORD).encode()).hexdigest()
        conn.execute(
            "INSERT INTO users (username, pw_hash, salt, role) VALUES (?,?,?,'admin')",
            (ADMIN_USERNAME, pw_hash, salt),
        )
        conn.commit()
    conn.close()


def _hash(pw: str, salt: str) -> str:
    return hashlib.sha256((salt + pw).encode()).hexdigest()


# ── Admin IP restriction ─────────────────────────────────────────────────────
# Один администратор на всю локальную сеть (/24 подсеть)


def _net24(ip: str) -> str:
    """Возвращает /24 подсеть IP."""
    if not ip:
        return ""
    parts = ip.split(".")
    if len(parts) == 4:
        return ".".join(parts[:3])
    return ip


def _admin_ip_allowed(ip: str) -> tuple:
    """
    Возвращает (allowed: bool, reason: str).
    Разрешает если нет ни одного активного admin, либо если
    запрашивающий IP в той же /24 подсети.
    """
    conn = get_conn()
    admins = conn.execute(
        "SELECT admin_ip FROM users WHERE role='admin' AND admin_ip IS NOT NULL AND admin_ip != ''"
    ).fetchall()
    conn.close()
    if not admins:
        return True, ""
    my_net = _net24(ip)
    for row in admins:
        admin_net = _net24(row["admin_ip"])
        if admin_net and my_net and admin_net == my_net:
            return True, ""
    return (
        False,
        "Администратор уже зарегистрирован в другой подсети. Только один администратор разрешён в локальной сети.",
    )


# ── Auth ─────────────────────────────────────────────────────────────────────


def register(username: str, password: str, ip: str = "") -> dict:
    salt = secrets.token_hex(16)
    pw_hash = _hash(password, salt)
    try:
        conn = get_conn()
        conn.execute(
            "INSERT INTO users (username, pw_hash, salt) VALUES (?,?,?)",
            (username.strip(), pw_hash, salt),
        )
        conn.commit()
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.close()
        return {"ok": True, "user_id": user_id, "username": username.strip()}
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Имя пользователя уже занято"}


def login(username: str, password: str, ip: str = "") -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, pw_hash, salt, role, admin_ip FROM users WHERE username=?",
        (username.strip(),),
    ).fetchone()
    conn.close()
    if not row:
        return {"ok": False, "error": "Пользователь не найден"}
    if _hash(password, row["salt"]) != row["pw_hash"]:
        return {"ok": False, "error": "Неверный пароль"}

    # Проверка IP для администратора
    if row["role"] == "admin" and ip:
        allowed, reason = _admin_ip_allowed(ip)
        if not allowed:
            return {"ok": False, "error": reason}
        # Запоминаем IP администратора при первом входе
        if not row["admin_ip"]:
            conn = get_conn()
            conn.execute("UPDATE users SET admin_ip=? WHERE id=?", (ip, row["id"]))
            conn.commit()
            conn.close()

    token = secrets.token_hex(32)
    expires = int(time.time()) + 86400 * 30
    conn = get_conn()
    conn.execute(
        "INSERT INTO sessions (token, user_id, expires, ip) VALUES (?,?,?,?)",
        (token, row["id"], expires, ip),
    )
    conn.commit()
    conn.close()
    return {
        "ok": True,
        "token": token,
        "user_id": row["id"],
        "username": username.strip(),
        "role": row["role"],
    }


def get_user_by_token(token: str):
    if not token:
        return None
    conn = get_conn()
    row = conn.execute(
        """SELECT u.id, u.username, u.role FROM users u
           JOIN sessions s ON s.user_id=u.id
           WHERE s.token=? AND s.expires>?""",
        (token, int(time.time())),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def logout(token: str):
    conn = get_conn()
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()
    conn.close()


# ── Queue ────────────────────────────────────────────────────────────────────


def queue_add(user_id: int, request_id: str, prompt: str, params: dict) -> int:
    conn = get_conn()
    conn.execute(
        "INSERT OR IGNORE INTO gen_queue (user_id, request_id, prompt, params) VALUES (?,?,?,?)",
        (user_id, request_id, prompt, json.dumps(params, ensure_ascii=False)),
    )
    conn.commit()
    pos = conn.execute(
        "SELECT COUNT(*) FROM gen_queue WHERE status='waiting' AND id <= (SELECT id FROM gen_queue WHERE request_id=?)",
        (request_id,),
    ).fetchone()[0]
    conn.close()
    return pos


def queue_status(request_id: str) -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, status, result, error FROM gen_queue WHERE request_id=?",
        (request_id,),
    ).fetchone()
    if not row:
        conn.close()
        return {"found": False}
    pos = conn.execute(
        "SELECT COUNT(*) FROM gen_queue WHERE status='waiting' AND id <= ?",
        (row["id"],),
    ).fetchone()[0]
    conn.close()
    d = dict(row)
    d["position"] = pos
    d["found"] = True
    return d


def queue_next_waiting():
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM gen_queue WHERE status='waiting' ORDER BY id ASC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def queue_set_processing(request_id: str):
    conn = get_conn()
    conn.execute(
        "UPDATE gen_queue SET status='processing', started=? WHERE request_id=?",
        (int(time.time()), request_id),
    )
    conn.commit()
    conn.close()


def queue_set_done(request_id: str, result: str):
    conn = get_conn()
    conn.execute(
        "UPDATE gen_queue SET status='done', result=?, finished=? WHERE request_id=?",
        (result, int(time.time()), request_id),
    )
    conn.commit()
    conn.close()


def queue_set_error(request_id: str, error: str):
    conn = get_conn()
    conn.execute(
        "UPDATE gen_queue SET status='error', error=?, finished=? WHERE request_id=?",
        (error, int(time.time()), request_id),
    )
    conn.commit()
    conn.close()


def queue_leave(request_id: str, user_id: int):
    conn = get_conn()
    conn.execute(
        "DELETE FROM gen_queue WHERE request_id=? AND user_id=? AND status='waiting'",
        (request_id, user_id),
    )
    conn.commit()
    conn.close()


def queue_cleanup(max_age_sec: int = 600):
    """Удаляем старые завершённые задачи."""
    cutoff = int(time.time()) - max_age_sec
    conn = get_conn()
    conn.execute(
        "DELETE FROM gen_queue WHERE status IN ('done','error') AND finished < ?",
        (cutoff,),
    )
    conn.commit()
    conn.close()


def queue_count_waiting() -> int:
    conn = get_conn()
    n = conn.execute(
        "SELECT COUNT(*) FROM gen_queue WHERE status='waiting'"
    ).fetchone()[0]
    conn.close()
    return n


# ── Models ───────────────────────────────────────────────────────────────────


def create_model(name: str) -> int:
    slug = slugify(name)
    conn = get_conn()
    try:
        conn.execute("INSERT INTO models (name, slug) VALUES (?,?)", (name, slug))
        conn.commit()
    except sqlite3.IntegrityError:
        slug = f"{slug}-{int(time.time())}"
        conn.execute("INSERT INTO models (name, slug) VALUES (?,?)", (name, slug))
        conn.commit()
    mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return mid


def get_model_by_id(model_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, name, slug, created FROM models WHERE id=?", (model_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_models() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, name, slug, created FROM models ORDER BY created DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def rename_model(model_id: int, name: str):
    conn = get_conn()
    conn.execute("UPDATE models SET name=? WHERE id=?", (name, model_id))
    conn.commit()
    conn.close()


def create_model_version(
    model_id: int, params: dict, vocab_size: int, notes: str = ""
) -> int:
    conn = get_conn()
    vnum = (
        conn.execute(
            "SELECT MAX(version_num) FROM model_versions WHERE model_id=?", (model_id,)
        ).fetchone()[0]
        or 0
    ) + 1
    conn.execute(
        "INSERT INTO model_versions (model_id, version_num, params, vocab_size, notes) VALUES (?,?,?,?,?)",
        (model_id, vnum, json.dumps(params, ensure_ascii=False), vocab_size, notes),
    )
    conn.commit()
    vid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return vid


def get_model_versions(model_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, version_num, params, vocab_size, best_epoch, best_val_loss, checkpoint_path, notes, created FROM model_versions WHERE model_id=? ORDER BY version_num DESC",
        (model_id,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("params"):
            try:
                d["params"] = json.loads(d["params"])
            except Exception:
                pass
        result.append(d)
    return result


def get_model_version_by_id(version_id: int):
    conn = get_conn()
    row = conn.execute(
        """SELECT mv.id, mv.version_num, mv.best_epoch, mv.best_val_loss,
                  mv.checkpoint_path, mv.params, mv.notes,
                  m.id as model_id, m.name as model_name, m.slug as model_slug
           FROM model_versions mv JOIN models m ON m.id=mv.model_id WHERE mv.id=?""",
        (version_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    if d.get("params"):
        try:
            d["params"] = json.loads(d["params"])
        except Exception:
            pass
    return d


def update_model_version(
    version_id: int, best_epoch=None, best_val_loss=None, checkpoint_path=None
):
    conn = get_conn()
    if best_epoch is not None:
        conn.execute(
            "UPDATE model_versions SET best_epoch=?, updated=strftime('%s','now') WHERE id=?",
            (best_epoch, version_id),
        )
    if best_val_loss is not None:
        conn.execute(
            "UPDATE model_versions SET best_val_loss=?, updated=strftime('%s','now') WHERE id=?",
            (best_val_loss, version_id),
        )
    if checkpoint_path is not None:
        conn.execute(
            "UPDATE model_versions SET checkpoint_path=?, updated=strftime('%s','now') WHERE id=?",
            (checkpoint_path, version_id),
        )
    conn.commit()
    conn.close()


def get_all_versions_flat() -> list:
    conn = get_conn()
    rows = conn.execute(
        """SELECT mv.id, mv.version_num, mv.best_epoch, mv.best_val_loss,
                  mv.checkpoint_path, mv.params, mv.notes,
                  m.id as model_id, m.name as model_name, m.slug as model_slug
           FROM model_versions mv JOIN models m ON m.id=mv.model_id
           ORDER BY m.created DESC, mv.version_num DESC"""
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("params"):
            try:
                d["params"] = json.loads(d["params"])
            except Exception:
                pass
        result.append(d)
    return result


def get_model_samples(version_id: int, epoch=None) -> list:
    conn = get_conn()
    if epoch is None:
        rows = conn.execute(
            "SELECT * FROM model_samples WHERE version_id=? ORDER BY epoch ASC, sample_num ASC",
            (version_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM model_samples WHERE version_id=? AND epoch=? ORDER BY sample_num ASC",
            (version_id, epoch),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_model_sample(version_id, epoch, sample_num, params, text, saved_name=None):
    conn = get_conn()
    conn.execute(
        "INSERT INTO model_samples (version_id, epoch, sample_num, params, text, saved_name) VALUES (?,?,?,?,?,?)",
        (
            version_id,
            epoch,
            sample_num,
            json.dumps(params, ensure_ascii=False),
            text,
            saved_name,
        ),
    )
    conn.commit()
    sid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return sid


def save_model_sample(sample_id: int, saved_name: str):
    conn = get_conn()
    conn.execute(
        "UPDATE model_samples SET saved_name=? WHERE id=?", (saved_name, sample_id)
    )
    conn.commit()
    conn.close()


def delete_model_sample(sample_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM model_samples WHERE id=?", (sample_id,))
    conn.commit()
    conn.close()


# ── Training epochs ───────────────────────────────────────────────────────────


def save_epoch(version_id, epoch_data: dict):
    conn = get_conn()
    conn.execute(
        """INSERT INTO training_epochs
           (version_id, epoch, train_loss, val_loss, train_ppl, val_ppl, lr, duration, is_best)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            version_id,
            epoch_data.get("epoch"),
            epoch_data.get("train_loss"),
            epoch_data.get("val_loss"),
            epoch_data.get("train_ppl"),
            epoch_data.get("val_ppl"),
            epoch_data.get("lr"),
            epoch_data.get("duration"),
            1 if epoch_data.get("is_best") else 0,
        ),
    )
    if epoch_data.get("is_best") and version_id:
        conn.execute(
            "UPDATE model_versions SET best_epoch=?, best_val_loss=? WHERE id=?",
            (epoch_data.get("epoch"), epoch_data.get("val_loss"), version_id),
        )
    conn.commit()
    conn.close()


def get_epochs_for_version(version_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM training_epochs WHERE version_id=? ORDER BY epoch ASC",
        (version_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_epochs_from_db() -> list:
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM training_epochs ORDER BY epoch ASC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        conn.close()
        return []


# ── Chats ─────────────────────────────────────────────────────────────────────


def get_chats(user_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        """SELECT c.id, c.title, c.updated, c.model_version_id,
                  mv.version_num, m.name as model_name
           FROM chats c
           LEFT JOIN model_versions mv ON mv.id=c.model_version_id
           LEFT JOIN models m ON m.id=mv.model_id
           WHERE c.user_id=? ORDER BY c.updated DESC""",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def create_chat(
    user_id: int, title: str = "Новый чат", model_version_id: int = None
) -> int:
    conn = get_conn()
    conn.execute(
        "INSERT INTO chats (user_id, title, model_version_id) VALUES (?,?,?)",
        (user_id, title, model_version_id),
    )
    conn.commit()
    cid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return cid


def delete_chat(chat_id: int, user_id: int):
    conn = get_conn()
    conn.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
    conn.commit()
    conn.close()


def rename_chat(chat_id: int, user_id: int, title: str):
    conn = get_conn()
    conn.execute(
        "UPDATE chats SET title=? WHERE id=? AND user_id=?", (title, chat_id, user_id)
    )
    conn.commit()
    conn.close()


def set_chat_model(chat_id: int, user_id: int, version_id: int):
    conn = get_conn()
    conn.execute(
        "UPDATE chats SET model_version_id=? WHERE id=? AND user_id=?",
        (version_id, chat_id, user_id),
    )
    conn.commit()
    conn.close()


def get_messages(chat_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, role, content, ts FROM messages WHERE chat_id=? ORDER BY ts ASC",
        (chat_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_message(chat_id: int, role: str, content: str) -> int:
    conn = get_conn()
    conn.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?,?,?)",
        (chat_id, role, content),
    )
    conn.execute("UPDATE chats SET updated=strftime('%s','now') WHERE id=?", (chat_id,))
    conn.commit()
    mid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return mid


# ── Corrections ───────────────────────────────────────────────────────────────


def add_word_correction(
    user_id: int, wrong: str, correct: str, context: str = "", msg_id: int = None
):
    conn = get_conn()
    conn.execute(
        "INSERT INTO word_corrections (user_id, msg_id, wrong, correct, context) VALUES (?,?,?,?,?)",
        (user_id, msg_id, wrong.strip(), correct.strip(), context),
    )
    conn.commit()
    conn.close()


def get_corrections(user_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT wrong, correct, context, ts FROM word_corrections WHERE user_id=? ORDER BY ts DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_corrections() -> dict:
    conn = get_conn()
    rows = conn.execute(
        "SELECT wrong, correct FROM word_corrections GROUP BY wrong ORDER BY MAX(ts) DESC"
    ).fetchall()
    conn.close()
    return {r["wrong"]: r["correct"] for r in rows if r["wrong"] not in {}}


def get_corrections_as_training_pairs() -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT wrong, correct FROM word_corrections ORDER BY ts ASC"
    ).fetchall()
    conn.close()
    return [(r["wrong"], r["correct"]) for r in rows]


# ── Datasets ──────────────────────────────────────────────────────────────────


def add_user_dataset(user_id: int, filename: str, filepath: str, words: int) -> int:
    conn = get_conn()
    conn.execute(
        "INSERT INTO user_datasets (user_id, filename, filepath, words) VALUES (?,?,?,?)",
        (user_id, filename, filepath, words),
    )
    conn.commit()
    did = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return did


def get_user_datasets(user_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, filename, words, ts FROM user_datasets WHERE user_id=? ORDER BY ts DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_dataset_paths() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT filepath FROM user_datasets").fetchall()
    conn.close()
    return [r["filepath"] for r in rows]


# ── Epoch management ──────────────────────────────────────────────────────────


def get_epoch_entry(version_id: int, epoch: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM training_epochs WHERE version_id=? AND epoch=?",
        (version_id, epoch),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def validate_epoch(version_id: int, epoch: int, custom_name: str) -> dict:
    conn = get_conn()
    conn.execute(
        "UPDATE training_epochs SET validated=1, custom_name=? WHERE version_id=? AND epoch=?",
        (custom_name.strip(), version_id, epoch),
    )
    conn.commit()
    conn.close()
    return get_epoch_entry(version_id, epoch) or {}


def delete_epoch_ckpt(version_id: int, epoch: int) -> dict:
    conn = get_conn()
    ver = conn.execute(
        "SELECT m.slug FROM model_versions mv JOIN models m ON m.id=mv.model_id WHERE mv.id=?",
        (version_id,),
    ).fetchone()
    conn.close()
    if ver:
        from core.config import model_ckpt_dir

        ckpt_file = model_ckpt_dir(ver["slug"]) / f"epoch_{epoch:04d}.pt"
        if ckpt_file.exists():
            ckpt_file.unlink()
    conn = get_conn()
    conn.execute(
        "UPDATE training_epochs SET ckpt_deleted=1 WHERE version_id=? AND epoch=?",
        (version_id, epoch),
    )
    conn.commit()
    conn.close()
    return {"ok": True, "epoch": epoch}


def get_epochs_for_chat(version_id: int, top_unvalidated: int = 3) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM training_epochs WHERE version_id=? ORDER BY val_loss ASC",
        (version_id,),
    ).fetchall()
    conn.close()
    result, unv = [], 0
    for r in rows:
        d = dict(r)
        if d["is_best"] or d.get("validated"):
            result.append(d)
        elif not d.get("ckpt_deleted") and unv < top_unvalidated:
            result.append(d)
            unv += 1
    return result


def get_all_epochs_rich(version_id: int) -> list:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM training_epochs WHERE version_id=? ORDER BY epoch ASC",
        (version_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_epoch_ckpt_path(version_id: int, epoch: int):
    conn = get_conn()
    ver = conn.execute(
        "SELECT m.slug FROM model_versions mv JOIN models m ON m.id=mv.model_id WHERE mv.id=?",
        (version_id,),
    ).fetchone()
    conn.close()
    if not ver:
        return None
    from core.config import model_ckpt_dir

    ckpt_dir = model_ckpt_dir(ver["slug"])
    best = ckpt_dir / "best_model.pt"
    if epoch == -1:
        return str(best) if best.exists() else None
    ep = ckpt_dir / f"epoch_{epoch:04d}.pt"
    if ep.exists():
        return str(ep)
    return str(best) if best.exists() else None


def get_models_with_versions() -> list:
    conn = get_conn()
    models = conn.execute(
        "SELECT id, name, slug, created FROM models ORDER BY created DESC"
    ).fetchall()
    result = []
    for m in models:
        md = dict(m)
        versions = conn.execute(
            "SELECT mv.id, mv.version_num, mv.best_epoch, mv.best_val_loss, mv.checkpoint_path, mv.notes, mv.params FROM model_versions mv WHERE mv.model_id=? ORDER BY mv.version_num ASC",
            (m["id"],),
        ).fetchall()
        vlist = []
        for v in versions:
            vd = dict(v)
            if vd.get("params"):
                try:
                    vd["params"] = json.loads(vd["params"])
                except Exception:
                    pass
            vd["chat_epochs"] = get_epochs_for_chat(vd["id"])
            vlist.append(vd)
        md["versions"] = vlist
        result.append(md)
    conn.close()
    return result
